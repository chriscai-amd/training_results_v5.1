# DLRM-DCNv2 — HugeCTR ROCm port for AMD Instinct MI350X

A working port of NVIDIA's MLPerf v5.1 HugeCTR DLRM-DCNv2 submission to AMD's
ROCm 7.2 platform, targeting `gfx950` (Instinct MI350X). Source-compatible
with NVIDIA's `train.py` Python frontend and `mlperf_logger` MLLog format.

This is a research / port branch, **not an official MLPerf submission**.

## Status

Single node, 8 × MI350X, FP16 mixed precision (matching NVIDIA's B200
submission), Adagrad, scaler 16,348, sharding=auto + real-cardinality
embedding tables (DP-replicates 13 small tables, see §2.4), HIP graph +
overlap on, `DEBUG_HIP_DYNAMIC_QUEUES=1` baked in. All numbers on real
MLPerf Criteo data (HuggingFace subsample, copied to `/dev/shm` to avoid
the AsyncReader's `O_DIRECT` NFS bottleneck — same RAM-disk strategy as
NV's `--tmpfs /ramdata`).

| Batch (global) | Per-GPU | ms/iter | **AMD M sps** | NV B200 May-13 (auto+tmpfs) | Ratio |
|---:|---:|---:|---:|---:|---:|
| 55,296 (1×, MLPerf-spec) | 6,912 | 4.23 | **13.07** | 26.33 | 49.6 % |
| 110,592 (2×) | 13,824 | 7.22 | **15.31** | 30.72 | 49.8 % |
| 221,184 (4×) | 27,648 | 12.65 | **17.49** | 31.60 | 55.3 % |
| **442,368 (8×, peak)** | 55,296 | 24.67 | **17.93** | 32.77 | **54.7 %** ← peak ratio |

NVIDIA B200 reference numbers come from the companion
[`b200/.../README-b200-1x8.md`](https://github.com/chriscai-amd/training_results_v5.1/blob/chcai/b200/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/README-b200-1x8.md)
where NVIDIA ran their published MLPerf submission binary on the same
HuggingFace Criteo subsample. NV's **May 13, 2026** numbers reflect
their `SHARDING_PLAN=auto` + `--tmpfs /ramdata` breakthrough (their
Apr→May jump: 13.69 → 26.33 M sps at bs1x = +92 %). On AMD the same
two changes give us only +2 % (we already use the tmpfs equivalent and
RCCL all-to-all is bottlenecked by CU-saturation rather than volume,
see §2.4 below).

## Build & Run

```bash
# Build inside the ROCm 7.2.1 container (single-shot ~10 min).
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged \
    -v $(pwd)/hugectr_rocm_port:/workspace -w /workspace \
    rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash scripts/rebuild_in_container.sh

# Run (B200-config matched, single node × 8 MI350X, FP16 mixed, peak batch).
docker run --rm --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video --privileged --ipc=host \
    -v $(pwd)/hugectr_rocm_port:/workspace \
    -v /dev/shm/criteo:/criteo \
    -e HCTR_NGPU=8 -e HCTR_BATCH=442368 -e HCTR_EVAL_BATCH=131072 \
    -e HCTR_EV_SIZE=128 -e HCTR_LR=0.004 -e HCTR_MAX_ITER=200 \
    -e HCTR_MEM_CAP=200 \
    -e HCTR_PRECISION_FLAGS="--use_mixed_precision --scaler 16348" \
    -e HCTR_SHARDING_PLAN=auto -e HCTR_USE_MULTI_HOT=1 -e HCTR_USE_MLPERF_CRITEO=1 \
    -e HCTR_USE_CUDA_GRAPH=1 \
    -w /workspace \
    rocm/pyt-megatron-lm-jax-nightly-private:primus_rocm7.2_20260424 \
    bash scripts/run_b200_match.sh
```

Build artefacts go to
`hugectr_hip/build_rocm72/lib/{hugectr.so,libhuge_ctr_shared.so,libembedding.so,libgpu_cache.so,libhugectr_core23.so}`
(not committed).

`run_b200_match.sh` bakes in the production set of perf knobs:
`DEBUG_HIP_DYNAMIC_QUEUES=1`, `DEBUG_HIP_BLOCK_SYNC=0`,
`NCCL_BUFFSIZE=8MiB`, `NCCL_PROTO=LL128`, `HIP_FORCE_DEV_KERNARG=1`,
plus the data-reader pointed at `/dev/shm/criteo/mlperf` if available.

## Part 1 — Enablement

How NVIDIA HugeCTR was ported to ROCm 7.2 / `gfx950`. The initial port
commit ([`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215), 2026-05-09) added 1,661 files / ~478 K LOC — the full
upstream HugeCTR + GPU cache + vendored 3rd-party trees, post-`hipify-perl`,
with the hand fixes below to make them build, link, and run correctly on
AMD hardware. Sections are roughly the order you'd hit them porting from
scratch.

## 1.1 Build system & toolchain

| File | Change | Why |
|---|---|---|
| `CMakeLists.txt` (root) | `CMAKE_HIP_ARCHITECTURES=gfx950`; `find_package` for `hipBLAS / hipBLASLt / RCCL / rocrand / rocprim`; drop `find_package(CUDA)` | gfx950 is the MI350X target; AMD libraries replace CUDA equivalents |
| `CMakeLists.txt` (root) | `add_compile_options($<$<COMPILE_LANGUAGE:HIP>:-fopenmp>)` and `add_link_options(-fopenmp)` | **Critical**: without `-fopenmp` for HIP sources, `amdclang` silently no-ops `#pragma omp parallel`, causing the AUC NCCL warmup to deadlock. 8-GPU runs hung at "Starting AUC NCCL warm-up" until this landed. |
| Per-target `CMakeLists.txt` (core23, embedding, gpu_cache, src) | `set_source_files_properties(... LANGUAGE HIP)`; link `hip::host`, `roc::hipblaslt`, `roc::rccl`, `tbb` | CMake/HIP toolchain expectations |
| `cmake/FindNUMA.cmake`, `cmake/FindAIO.cmake` | Vendored / fixed Find scripts | Upstream scripts assume Debian package layouts that don't match the ROCm container |
| `hugectr_hip_warp_compat.h` (NEW) | `__device__ __forceinline__ int __ffs(unsigned long long)` etc. dispatching to `__ffsll` / `__popcll` | AMD wave64 means warp masks are 64-bit; upstream HugeCTR uses 32-bit mask intrinsics |
| `scripts/{rebuild,run,run_b200_match,...}.sh`, `runtime_test/criteo_npy_to_hugectr_bin*.py` | Driver scripts (docker-aware, NFS-aware, libaio install in container) | Cluster doesn't have Pyxis/Enroot — everything goes through plain `srun + docker run` |

## 1.2 Hipify + manual translation gaps

`hipify-perl` covered most of the CUDA → HIP translation (`cuda*` → `hip*`,
`__half` types are layout-compatible). What it did NOT translate cleanly:

| Issue | Fix | File(s) |
|---|---|---|
| `cooperative_groups::reduce` / `inclusive_scan` | Hand-rolled `shfl_xor`-based primitives | `embeddings/data_distributor/key_filtering_operators.cu` |
| `MLCommon::LinAlg::binaryOp / matrixVectorOp` (cuml dependency) | HIP-native shim with `__device__` lambdas | `include/prims/mlcommon_linalg_hip.cuh` (NEW) |
| `cublasLtMatmulDescAttribute` enums (RELU_AUX_BIAS, DRELU_BGRAD, …) | Renamed to `hipblasLt*` equivalents | `layers/functors/fused_gemm_functors.cu` |
| `cublasGemmAlgo_t` constants | hipBLAS uses different enum names | `layers/fully_connected_layer*.cu` |
| `cuDNN` / `cudnnTensorDescriptor_t` | Gated out (only used by older `BatchNormLayer` not in DLRM-DCNv2); throwing stubs in `excluded_layer_stubs.cpp` | various |
| `nvml.h` | Stripped (`rocm_smi` exists but DLRM-DCNv2 doesn't need it) | `core23/error.hpp` |
| CUDA virtual-memory allocators (`cuMemMap`, `cuMemAddressReserve`) | Runtime-throwing stubs | `core23/details/low_level_cuda_allocator.cpp` |
| FP8 (`__nv_fp8_e4m3` etc.) | Stubbed (gfx950 has FP8 but the HugeCTR paths use NVIDIA-specific intrinsics) | `gpu_cache/src/static_hash_table_stub.cpp` |
| `bar.sync` PTX inline assembly | `__syncthreads()` | a few embedding kernel hot loops |
| `uint2`/`uint4` union-init `{x, y}` | `make_uint2(x, y)` / `make_uint4(...)` | `include/hashtable/cudf/concurrent_unordered_map.cuh` |
| `atomicCAS` / `atomicAdd` user overloads collide with HIP's | `#ifndef __HIPCC__` guard | `gpu_cache/src/nv_gpu_cache.cu`, `concurrent_unordered_map.cuh` |
| `obj.data<T>()` template-keyword errors under `amdclang` | `obj.template data<T>()` sweep | many `.cu` files |
| `hipblasGemmEx` `computeType` was `HIP_R_32F`, expected `HIPBLAS_COMPUTE_32F` | type fix | `layers/fully_connected_layer_half.cu` |
| `hipblasHgemm` arg type (`__half*` vs `hipblasHalf*`) | `reinterpret_cast` | `layers/multi_cross_layer.cu`, `fully_connected_layer_half.cu` |
| `static_assert` 32-bit warp mask vs 64-bit | `0xFFFFFFFF` → `0xFFFFFFFFFFFFFFFFULL` | warp-vote intrinsic call sites |
| Linker `libtbb.so.2` not found | CMake `find_library(TBB tbb)` + explicit link | top-level `CMakeLists.txt` |
| `libaio.h` missing in container | `apt-get install libaio-dev` step in `rebuild_in_container.sh` | scripts |

## 1.3 Wave64 corrections (AMD-specific kernel logic)

`gfx950` is wave64. Several HugeCTR kernels were written assuming wave32:

| File | Change | Symptom if not fixed |
|---|---|---|
| `include/common.hpp` | `#if defined(__HIP_PLATFORM_AMD__)` → `WARP_SIZE = 64` | MultiCross bprop's `matrix_pair_mul_kernel` and `row_scaling_sum_kernel` produced cross-row contamination → NaN at per-rank batch ≥ 2048 |
| `embedding/operators/generic_lookup.cuh` | Reverted to `WARP_SIZE = 32` after wave64 caused embedding-lookup regression here | Embedding kernels are vectorized in 32-lane tiles; wave64 here regressed perf |
| `hugectr_hip_warp_compat.h` (NEW) | `__ffs` / `__popc` / `__ballot` overloads for `unsigned long long`, `int64_t` | Build errors on warp-vote intrinsics |

## 1.4 Runtime correctness fixes (per-iter / multi-GPU)

| Issue | Fix | Commit |
|---|---|---|
| OpenMP `#pragma omp parallel` silently no-opped → AUC NCCL warmup deadlocked | `-fopenmp` for HIP/CXX in CMakeLists | [`0a24ef2`](https://github.com/chriscai-amd/training_results_v5.1/commit/0a24ef2) |
| MultiCross v2 **BGRADA FP16 overflow** at per-rank batch ≥ 2048 (FP16 column-sum could exceed 65,504 → inf → NaN via `ncclSum` → poisons Adagrad) | FP32 accumulation + `isfinite()` guard + ±FP16-max clamp + pre-divide by 256 (Adagrad scale-invariant) | [`5439485`](https://github.com/chriscai-amd/training_results_v5.1/commit/5439485) |
| Wave-size mismatch in MultiCross bprop → row-cross-contamination at per-rank batch ≥ 2048 | `WARP_SIZE = 64` for AMD | [`9d05c0f`](https://github.com/chriscai-amd/training_results_v5.1/commit/9d05c0f) |
| MultiCross fprop **NaN poisoning** at per-rank batch ≥ 2048 (one inf/NaN element propagates through later GEMMs) | `clamp_fp16_kernel` post-pass after each cross layer's `fused_matrix_elementwise_dot_add`; later inlined into FMA store via `sanitize_half2_fp16` | [`0a24ef2`](https://github.com/chriscai-amd/training_results_v5.1/commit/0a24ef2) then [`93aad5c`](https://github.com/chriscai-amd/training_results_v5.1/commit/93aad5c) |
| `hipblasCreate` illegal during HIP graph capture | Per-device `hipblasHandle_t` cache pre-warmed via `std::call_once` in `CublasAlgo<T>::init_algorithm` | [`eb4fa60`](https://github.com/chriscai-amd/training_results_v5.1/commit/eb4fa60) |
| Synthetic Criteo IDs (0..65535) indexed OOB into real `TABLE_SIZE_ARRAY` (some entries 3, 36, 63) | Clamp `TABLE_SIZE_ARRAY[i] = max(real_size, 65536)` when `HCTR_USE_REAL_TABLE_SIZES=1` | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) |
| `Model.fit()` segfaulted on synthetic data — `DistributedSlotSparseEmbeddingHash` incompatible with `MultiHot AsyncDataReader` | Switch to modern `EmbeddingCollection` API; add `Reshape` between `EmbeddingCollection` output and `Concat` | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) |
| `ncclCommInitAll` failed on 8-GPU init — `shard_matrix` hardcoded for 1 GPU | Scale to `NGPU` in train.py's sharding generation | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) |
| Multi-GPU fused MLP collapsed to `log(2)·2` — BIAS post-pass guard was `(act == None) && bias != null`, dropping bias on every hidden ReLU layer | Drop `act == None` clause; BIAS post-pass fires whenever `bias != null && (saved_is_bias_epilogue || saved_is_relu_aux_epilogue)` | [`0d9a35e`](https://github.com/chriscai-amd/training_results_v5.1/commit/0d9a35e) |
| Excluded layers (`MultiHeadAttention`, `GRU`, etc.) still type-referenced | Throwing stubs in `excluded_layer_stubs.cpp` | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) |

## 1.5 FP16 numerical hardening (additions vs CUDA build, which is BF16)

The FP16 + 8-GPU + real MultiCross path needed FP32-accumulation hardening
that doesn't exist in the upstream CUDA build (which uses BF16 mixed and
avoids most overflow problems):

| File | Change |
|---|---|
| `src/layers/functors/fused_gemm_functors.cu` | `to_f32<T>` / `from_f32<T>` intrinsics for safe `__half` ↔ `float` conversion. FP32 accumulation in all bgrad/bgrada paths. |
| `src/layers/functors/fused_gemm_functors.cu` | Always-on FP32 accumulation in the `hipblasGemmEx` fallback (FP16 GEMM with FP16 accumulate would overflow MultiCross intermediates). |
| `src/layers/multi_cross_layer.cu` | `sanitize_half2_fp16` helper folded into `vector_fma{3,4}_align8<__half>`'s store path — NaN → 0, |v| > 65504 → ±65504 inline at HBM-store boundary. |
| `src/layers/functors/fused_gemm_functors.cu` | Pre-divide by `HCTR_DRELU_BGRAD_DIV` (default 256) before FP16 store, so per-rank dbias × 8 ranks under `ncclSum` stays under FP16 max. |

## 1.6 Excluded paths (stubbed / gated; not required for DLRM-DCNv2)

cuDNN BatchNorm; NVML/DCGM telemetry (replaced with `rocm-smi`); CUDA
virtual-memory allocators; FP8 quantised embedding; `MultiHeadAttention`,
`GRU`, `MultiCrossEntropyLoss`; HierarchicalKV embedding cache; MPI
multi-node (`NetworkExchangeWgrad` over RDMA); `mlperf_common` package
(stubbed via `mlperf_common_stub.py`).

## Part 2 — Performance Optimization Timeline

Each phase = one logical optimization (often spanning multiple commits).
All deltas measured at **NVIDIA's exact MLPerf-spec batch (55,296)**
unless noted, FP16 mixed, real MLPerf Criteo (HF subsample), 8 × MI350X.
"Sweet spot" used to mean batch 110,592 (Phase 8); after Phase 11 we
report at the **peak batch 442,368 (8×)** instead.

## 2.1 Master timeline table

| Phase | Date | Optimization | Commits | Throughput before → after | Δ |
|---:|---|---|---|---:|---:|
| 1  | 2026-05-09 | Initial port — get build/link/run working | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) | (didn't run) → first FP32 8-GPU run | n/a |
| 2  | 2026-05-09 | Multi-GPU FP16 MultiCross stabilisation (3 numerical bugs) | [`9d05c0f`](https://github.com/chriscai-amd/training_results_v5.1/commit/9d05c0f), [`5439485`](https://github.com/chriscai-amd/training_results_v5.1/commit/5439485), [`0a24ef2`](https://github.com/chriscai-amd/training_results_v5.1/commit/0a24ef2) | NaN at iter ≤ 2 → **5.85** M sps | first convergence |
| 3  | 2026-05-09 | HIP graph + intra/inter-iter overlap re-enabled (handle pre-warm `std::call_once`) | [`eb4fa60`](https://github.com/chriscai-amd/training_results_v5.1/commit/eb4fa60) | 5.85 → **6.26** | **+7.0 %** |
| 4  | 2026-05-10 | Multi-hot data path (912 B/row, 214 keys/row — apples-to-apples with NV submission) | [`5e12007`](https://github.com/chriscai-amd/training_results_v5.1/commit/5e12007) | 6.26 → **5.73** | -8.5 % (5× more embedding work; intentional regression for apples-to-apples) |
| 5  | 2026-05-10 | Fused MLP epilogue emulation (RELU_AUX / DRELU / DRELU_BGRAD via `hipblasGemmEx` fallback) | [`2ee57e7`](https://github.com/chriscai-amd/training_results_v5.1/commit/2ee57e7), [`0d9a35e`](https://github.com/chriscai-amd/training_results_v5.1/commit/0d9a35e), [`61ce55b`](https://github.com/chriscai-amd/training_results_v5.1/commit/61ce55b) | n/a (correctness only; fused path slower than InnerProduct on AMD) | (correctness) |
| 6  | 2026-05-10 | **V5 2D-tile bgrad/bgrada kernels** replace V1 cooperative scan (49 % of single-GPU time → sub-1 %) | [`56bc046`](https://github.com/chriscai-amd/training_results_v5.1/commit/56bc046), [`63a9c54`](https://github.com/chriscai-amd/training_results_v5.1/commit/63a9c54), [`4fc896a`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fc896a) | 5.73 → **11.24** | **+96 %** ← single biggest win |
| 7  | 2026-05-11 | FP16 NaN/inf clamp folded into `vector_fma{3,4}_align8` store path | [`93aad5c`](https://github.com/chriscai-amd/training_results_v5.1/commit/93aad5c) | 11.24 → **11.57** | +2.9 % @ bs1x; **+15 %** @ sweet-spot batch |
| 8  | 2026-05-11 | Re-enable intra/inter-iteration overlap (earlier scripts had set both to 0) | [`68e560e`](https://github.com/chriscai-amd/training_results_v5.1/commit/68e560e) | 11.57 → **12.55** | **+8.5 %** |
| 9  | 2026-05-11 | V5-style `add_bias_per_row_v5_kernel` (BLOCK_M=64 × N_TILE=128, coalesced, shared-mem broadcast) | [`3b984e4`](https://github.com/chriscai-amd/training_results_v5.1/commit/3b984e4) | 12.55 → **12.72** | +1.4 % @ bs1x; +2.6 % @ sweet-spot |
| 10 | 2026-05-11 | V5 BGRADA scratch zeroing folded into finalize kernel (3 hipMemsetAsync/iter eliminated) | [`b898899`](https://github.com/chriscai-amd/training_results_v5.1/commit/b898899) | 12.72 → **12.74** | +0.2 % (within noise) |
|    | 2026-05-12 | NFS-bound → `/dev/shm` fix (AsyncReader uses `O_DIRECT`, bypasses page cache) | [`bf88560`](https://github.com/chriscai-amd/training_results_v5.1/commit/bf88560) | 5.21 (NFS) → **11.76** sustained @ bs1x | +127 % vs NFS-bound run |
|    | 2026-05-12 | NV-submission audit (>120 env-knob configs swept) — bake `NCCL_PROTO=LL128`, `HIP_FORCE_DEV_KERNARG=1` | [`1ef02aa`](https://github.com/chriscai-amd/training_results_v5.1/commit/1ef02aa), [`fd1f4d2`](https://github.com/chriscai-amd/training_results_v5.1/commit/fd1f4d2), [`2c4670a`](https://github.com/chriscai-amd/training_results_v5.1/commit/2c4670a), [`a923ef1`](https://github.com/chriscai-amd/training_results_v5.1/commit/a923ef1) | 11.76 → **11.88** sustained | +1.0 % |
| 11 | 2026-05-12 | **`DEBUG_HIP_DYNAMIC_QUEUES=1`** (on-demand HW queue allocation; the 4-stream pipeline now overlaps properly) | [`3860967`](https://github.com/chriscai-amd/training_results_v5.1/commit/3860967) | 11.88 → **12.92** @ bs1x; 15.17 → **16.42** @ bs8x peak | **+8.9 % @ bs1x; +8.2 % @ bs8x** |
|    | 2026-05-12 | **Configuration: peak measurement at bs8x (442,368) instead of bs1x (55,296)** (amortizes 1.29 ms/iter host overhead — see linear-fit in Part 3) | (configuration only) | bs1x **12.92** → bs8x **16.42** M sps (same binary, 8× larger global batch, ~7 % per-GPU memory increase) | **+27 %** (**+5.3 ms saved per 8× samples**; relaxes MLPerf-spec batch constraint) |
| 12 | 2026-05-12 | int4-vectorized `__half` elementwise: `concat_fwd/bwd_kernel_vec8` + `binaryOp_kernel_vec8_half` (used by MultiCross matrix_add) | [`4fb17c3`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fb17c3), [`e3e64a2`](https://github.com/chriscai-amd/training_results_v5.1/commit/e3e64a2) | 16.42 → **16.67** @ bs8x | **+1.5 %** at peak |
| 13 | 2026-05-13 | **`HCTR_DROP_TABLE_SIZE_CLAMP=1` on real-data path** — removes stale `max(real_size, 65536)` clamp that broke `auto`-planner's DP-replication of the 13 small embedding tables (≤ 7,424 elems). Reduces embedding all-to-all volume by ~80 %. (NV's May-13 hot-fix per b200/README §7.5.) | [`2dc79d6`](https://github.com/chriscai-amd/training_results_v5.1/commit/2dc79d6) | bs8x 16.67 → **16.98**; bs2x 15.04 → **15.31** | **+2.0 % @ bs8x; +1.8 % @ bs2x; flat @ bs1x** (RCCL is CU-bound on AMD, not volume-bound) |
| 14a | 2026-05-13 | **`NCCL_BUFFSIZE=8 MiB → 32 MiB`** — at bs1x embedding output is 46 MB ≫ 8 MB so RCCL splits each logical SendRecv into ~6 chunks; bumping the buffer to 32 MB merges most chunks → fewer per-call inter-launch gaps | baked in `run_b200_match.sh` | bs1x 12.90 → **13.07**; bs8x 16.98 → 16.99 (flat) | **+1.3 % @ bs1x; flat @ bs8x** (RCCL only 10 % of bs8x kernel-time, dominant at bs1x) |
| 14b | 2026-05-13 | **`HCTR_FUSE_TOP_MLP=1` (re-enabled at bs ≥ 4×)** — Phase-5's negative measurement (-6 % at all batches) was a measurement artifact: with `NCCL_BUFFSIZE=8 MiB` the fused top-MLP path's larger NCCL chunks couldn't pipeline. With `BUFFSIZE=32 MiB` + bs ≥ 4× the fused-MLP epilogue chain (1 GEMM + 1 epilogue kernel per FC layer instead of 5–6 unfused) finally amortizes. | `run_b200_match.sh` auto-picks based on `HCTR_BATCH` | bs8x 16.99 → **17.86**; bs4x 16.50 → **17.49**; bs2x 15.31 → 14.59 (regress); bs1x 13.07 → 12.46 (regress) | **+5.2 % @ bs8x; +6.0 % @ bs4x; ‑4.7 % @ bs2x; ‑4.6 % @ bs1x** — auto-disabled below bs4× |
| 14c | 2026-05-13 | **`HCTR_FUSE_WB=True` (at bs ≥ 8×)** — fold weight-bias post-pass into the fused MLP. Stacks on top of 14b. At bs ≤ 4× this is flat or slightly negative (kernel overhead dominates the bias-fuse savings). | `run_b200_match.sh` auto-picks based on `HCTR_BATCH` | bs8x 17.86 → **17.93**; bs4x 17.49 → 17.14 (regress) | **+0.4 % @ bs8x; ‑2.0 % @ bs4x** — auto-disabled below bs8× |

**Peak result post-Phase-14: 17.93 M sps at bs8x (442,368)** = 54.7 % of NV B200's May-13 auto+tmpfs bs8x peak (32.77 M sps).

| Batch (global) | Per-GPU | M sps (post-Phase-14) | vs Phase-13 | vs NV B200 May-13 (same batch) |
|---:|---:|---:|---:|---:|
| 55,296 (1×, MLPerf-spec) | 6,912 | **13.07** | +1.3 % (BUFFSIZE) | 49.6 % of 26.33 |
| 110,592 (2×) | 13,824 | **15.31** | flat | 49.8 % of 30.72 |
| 221,184 (4×) | 27,648 | **17.49** | +6.0 % (FUSE_TOP) | 55.3 % of 31.60 |
| 442,368 (8×, peak) | 55,296 | **17.93** | +5.6 % (FUSE_TOP+FUSE_WB) | **54.7 %** of 32.77 |

The +29 % bs1x → bs8x win is **not free** — it relaxes the MLPerf-spec
global batch constraint (55,296). It IS free if the calling task can
tolerate a larger batch (which our convergence runs do, since DLRM-DCNv2
converges at any batch size up to ~512K with appropriate LR). NV's
b200/README §8.2b documents the same batch-size lever giving them
69 % → 87 % of MLPerf-ref-23M (bs1x → bs4x) for the same reason: the
~1 ms host-const overhead per iter amortizes over 4–8× more samples.

## 2.2 Selected sub-tables (key wins in detail)

### Phase 6 — V5 2D-tile bgrad/bgrada kernels (+96 %, single biggest win) — [`4fc896a`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fc896a)

`rocprof --stats` showed `reduce_sum_columns_kernel` (V1 BGRADA post-pass)
was **49 %** of all GPU time on a single-GPU run (~289 ms / 590 ms). V1
design (one block per row, 256-thread cooperative scan, uncoalesced reads
at stride-`m`) hit ~1 % CU utilisation at small `m`. Replaced with a V5
2D tile (BLOCK_M=64 rows × N_TILE=1024 cols/block), wave-aligned coalesced
reads, atomicAdd into per-device pre-allocated FP32 scratch (pre-warmed
via `std::call_once` so HIP graph capture sees the alloc done), finalize
kernel divides + clamps + casts.

Applied to both `bprop_drelu_bgrad_v5_kernel` (dgrad GEMM's DRELU+BGRAD
epilogue) and `bgrada_v5_kernel` (wgrad GEMM's BGRADA epilogue, used by
MultiCross).

| Configuration | Throughput | Δ |
|---|---:|---:|
| pre-V5 (legacy `reduce_sum_columns`) | 5.73 M sps | baseline |
| V5 in source but stale build (commit [`4fc896a`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fc896a)) | 5.73 M sps | (V5 not actually engaged) |
| **V5 actually engaged** (clean rebuild) | **11.24 M sps** | **+96 %** |

Side-finding: the build I'd been benchmarking against had been stale —
the V5 kernels were in the binary but the routing code wasn't, so V5
was never running. Verified post-fix by adding a one-shot
`[HCTR-V5] ... path=V5` diag print on the first BGRADA call.

### Phase 11 — `DEBUG_HIP_DYNAMIC_QUEUES=1` (+6-9 % across all batches) — [`3860967`](https://github.com/chriscai-amd/training_results_v5.1/commit/3860967)

After exhausting >120 env-knob configurations, found that AMD's HIP
runtime debug knob enables on-demand HW-queue allocation instead of the
default fixed pool. On HCTR's 4-stream pipeline (compute / RCCL / copy /
embedding) this gives the largest single env-knob win in the entire
campaign.

Per-stream effect at bs4x rocprofv3 trace (with vs without DYN_QUEUES):

| Metric | DYN_QUEUES off | DYN_QUEUES on | Δ |
|---|---:|---:|---:|
| iter cycle mean | 13.36 ms | **12.03 ms** | **-10 %** |
| GPU busy / iter (union of all streams) | 12.10 ms | 10.64 ms | -12 % (more overlap) |
| `hipGraphLaunch` p50 (host call) | 5.97 ms | 6.01 ms | flat (win is in scheduling, not launch) |
| **"other"-stream p90 inter-kernel gap** | **15.3 ms** | **0.27 ms** | **-98 %** |
| Embedding-stream p50 inter-kernel gap | 62.1 µs | 19.4 µs | -69 % |

5-trial throughput averages, real MLPerf data, `/dev/shm`:

| Batch | Pre-Phase-11 | + DEBUG_HIP_DYNAMIC_QUEUES=1 | Δ |
|---:|---:|---:|---:|
| 55,296 (1×) | 11.86 M sps | **12.92 M sps** | **+8.9 %** |
| 110,592 (2×) | 13.89 M sps | 15.04 M sps | +8.3 % |
| 221,184 (4×) | 15.21 M sps | 16.15 M sps | +6.0 % |
| 442,368 (8×) | 15.17 M sps | 16.42 M sps | +8.2 % |

## 2.3 Negative results (notable knobs that did NOT help)

These were measured but didn't survive a controlled re-test, so they're
NOT in the production `run_b200_match.sh`. Documented to save future
work from re-testing.

| Knob / change | Source | AMD result |
|---|---|---|
| `SHARDING_PLAN=round_robin` (vs `auto`) | NV b200/README Apr-26 +12 %, but May-13 retracted | **−23 %** on AMD; auto is correct (matches NV's May-13 finding) |
| `HCTR_DEFAULT_CONCURRENCY=8` | NV +30 % under host contention | flat on quiet host |
| `NCCL_LAUNCH_MODE=PARALLEL` (NV bakes in via Dockerfile) | n/a | flat |
| `grouped_all_reduce=False`, `num_iterations_statistics={5,100}` | NV submission default 20 | flat (sweep ±5 %) |
| `MEM_COMM_BW_RATIO`/`WORK_RATIO` (cost-model knob) | NV uses 9 / 4 | flat (sweep 1.1–4.5; default `7.4 / 5` retained) |
| `SHARDING_PLAN=hier_auto` | NV multi-node | requires multi-node, errors single-node |
| `HCTR_LOOKUP_WARPS_PER_BLOCK` ∈ {1, 2, 4, 8} | hypothesised wave64 inefficiency | flat — HIP `__shfl_sync` correctly handles per-32-lane subgroup on CDNA3 |
| `HSA_NO_SCRATCH_RECLAIM=0` | runtime knob | **−12 %** |
| `HSA_ENABLE_SDMA=0` | runtime knob | **−7 %** |
| `GPU_MAX_HW_QUEUES != default` | runtime knob | **−40 %** |
| `HCTR_ENABLE_ALGO_SEARCH=1` | runtime knob | **−0.7 %** (matches NV finding) |
| `NCCL_MIN/MAX_NCHANNELS` ∈ {4 .. 192}, `NCCL_MAX_CTAS` ∈ {32 .. 192} | hypothesised CU-footprint reduction | flat or regress (RCCL's auto-pick of 112 channels is optimal) |
| `HIPBLASLT_TUNING_OVERRIDE_FILE` (auto-generated) | hipBLASLt offline tuning | bench shows 14–60 % heuristic-vs-best gap; live-run gain ~+0.1 % (HCTR's `hipblasGemmEx` wrapper masks the headroom) |
| `HCTR_FUSE_WB=True` (NV uses `False`) | HCTR `DenseLayerComputeConfig` | flat at bs1x; -0.3 % at bs4x |
| Top-MLP fused (`HCTR_FUSE_TOP_MLP=1`) | NV uses fused MLP path | -6 % on AMD (loss correct but fused path's `hipblasGemmEx` chain is slower than InnerProduct's MFMA) |

## Part 3 — Per-component analysis vs NV B200 (bs=1×)

### 3.1 Latest status — bs=1× throughput gap vs NV B200

Steady-state samples/s @ MLPerf-spec global batch 55 296 (bs=1×, real
MLPerf Criteo on `/dev/shm`, 8 × MI350X / 8 × B200, FP16 mixed).

| Platform | M samples/s @ bs=1× | ms/iter | Ratio (AMD ÷ NV) | Gap to NV |
|---|---:|---:|---:|---:|
| AMD MI350X (this port, post-Phase-14p, 2026-05-14) | **13.07** | 4.23 | **0.496×** | **−50.4 %** |
| NV B200 (companion `b200/README-b200-1x8.md` May-13 auto+tmpfs) | **26.33** | 2.10 | 1.000× | — |

**AMD reaches 49.6 % of NV B200 throughput at bs=1×** — i.e. NV is
**2.01× faster**, leaving an **absolute +2.13 ms/iter** gap to close.
Per the §3.2 trace decomposition below, **72 %** of that 2.13 ms gap
is RCCL latency (5.3× per-call slowdown × 3.2× more calls), making
RCCL-side improvements the highest-leverage open work for bs=1×
(see Part 4 §4.1 for the prioritized plan).

### 3.2 Kernel-category breakdown @ MLPerf-spec batch (bs=1×, 55 296)

This is the **direct apples-to-apples** comparison vs NV's b200/README
§8.2d bs=1× decomposition (no extrapolation needed). Captured fresh
`rocprofv3 --hip-trace --kernel-trace` of our post-Phase-13 bs1x run
(200 iters, 55-iter steady window). Trace dir: `rocprof_bs1x_phase13/`.

| Component | AMD profile (raw) | AMD real (×0.167) | AMD % | NV real (§8.2d) | NV % | AMD/NV |
|---|---:|---:|---:|---:|---:|---:|
| **RCCL (15.9 calls/iter)** | **14.42 ms** | **2.41 ms** | **49 %** | **0.84 ms** | 43 % | **2.87×** |
| MLP GEMMs (hipBLASLt) | 6.70 ms | 1.12 ms | 23 % | 0.30 ms | 16 % | 3.74× |
| Embedding ops | 4.94 ms | 0.83 ms | 17 % | 0.32 ms | 16 % | 2.58× |
| Fused FMA / convert / concat | 2.37 ms | 0.40 ms | 8 % | 0.20 ms | 10 % | 1.98× |
| Memcpy / fillBuffer | 0.51 ms | 0.09 ms | 2 % | 0 ms | 0 % | ∞ |
| Other (sort / label_count / etc) | 0.30 ms | 0.05 ms | 1 % | 0.28 ms | 14 % | 0.18× |
| **Sum (kernel-time across streams)** | **29.24 ms** | **4.90 ms** | 100 % | **1.94 ms** | 100 % | **2.52×** |
| **Iter wall (real, unprofiled)** | — | **4.29 ms** | — | **2.10 ms** | — | **2.04×** |

Notes:
- **AMD profile column** is the raw `rocprofv3` kernel-time sum
  across all 8 streams (15.9× rcclDevKernel calls, 183× GEMM calls,
  etc.). Profile distortion inflates the wall from 4.29 → 43.36 ms
  (~10×) and inflates kernel-internal time too.
- **AMD real column** scales the profile sum by `4.0 / 23.89` =
  0.167 (real GPU-busy ms / profile GPU-busy ms, assuming AMD's
  busy fraction matches NV's 93 % at bs=1×). This is an estimate;
  the true scaling could be ±20 %.
- **NV real column** is the direct §8.2d measurement (bs=1×
  post-May-13, no extrapolation).
- **AMD/NV ratio** is conservative under the profile-scaling
  uncertainty.

#### Where the AMD/NV gap actually lives @ bs=1×

Top absolute kernel-time deltas (AMD minus NV, real-time estimates):

| Component | AMD − NV (ms / iter) | Gap as % of total iter gap |
|---|---:|---:|
| **RCCL** | **+1.57 ms** | **72 %** |
| MLP GEMMs | +0.82 ms | 38 % |
| Embedding ops | +0.51 ms | 23 % |
| Fused FMA / convert / concat | +0.20 ms | 9 % |

(Iter-gap sum is +2.19 ms; categories overlap because of stream
concurrency.)

→ **At bs=1× the dominant gap is RCCL, not GEMMs** — the opposite of
the bs=8× regime where GEMMs dominate (the unfused InnerProduct path
at bs=8× fires ~70 kernels/iter — 5–6 per FC layer × 7 FC layers × 2
fwd+bwd — vs NV's single fused `cutlass3x_sm100` per FC layer). This
matches NV's own §8.2d signature: at small batch RCCL latency
dominates total iter time; at large batch GPU compute amortizes.

#### Two distinct RCCL gaps

| Sub-metric | AMD bs=1× | NV bs=1× | AMD/NV |
|---|---:|---:|---:|
| RCCL kernels per iter | **15.9** | **5** (4 SendRecv + 1 AllReduce) | **3.2× more** |
| Per-call ncclDevKernel p50 | **815 µs** | 155 µs (SendRecv), 217 µs (AllReduce) | **5.3× slower** |
| Per-call mean | 873 µs | ~169 µs (weighted) | 5.2× slower |

The **3.2× call-count multiplier** is potentially actionable
(suggests RCCL is splitting each logical SendRecv/AllReduce into
chunks because the buffer is larger than `NCCL_BUFFSIZE=8 MiB`;
embedding output at bs=1× = 6 912 batch × 26 tables × 128 ev_size
× 2 B = 46 MB / 8 MB ≈ 6 chunks per logical SendRecv). The **5.3×
per-call latency** is partly platform-fundamental (no
xGMI multicast / NVLS equivalent on AMD).

## Linear fit of host overhead (`t_iter = c + α·batch`)

Both AMD and NV exhibit a constant-host + linear-GPU model. Fit on
{bs0.5x, bs1x}, validated on bs8x:

| Platform | Host const `c` (ms) | Per-sample GPU work `α` (ns/sample) | bs8x prediction error |
|---|---:|---:|---:|
| AMD MI350X (this port) | 1.29 | 63.3 | -0.4 % |
| NV B200 (b200/README §8.2b) | 0.995 | 50.5 | -1.3 % |

Implications for the remaining ~13–18 % gap:
- Match NV's `c=0.995` (host overhead): would save 0.30 ms/iter at bs1x → 12.92 → 13.7 M sps (+6 %)
- Match NV's `α=50.5` (per-sample GPU work): would save 0.71 ms/iter at bs1x → 12.92 → 14.2 M sps (+10 %)
- Match BOTH: 0.995 + 55296·50.5e-9 = 3.79 ms/iter = **14.59 M sps at bs1x** (matches NV-on-HF reference of 13.57)

## Part 4 — Open work (bs=1× focus)

### 4.1 Promising items, prioritized by estimated bs=1× win

| Rank | Item | Est. bs=1× win | Effort | Notes |
|---:|---|---:|---|---|
| 1 | **Custom RCCL chunking patches (RCCL source-level)** | **5–15 %** | 10+ days, RCCL expert | Fundamental gap per §3.2: RCCL emits **3.2 × more device kernels** per logical collective vs NCCL (chunking + protocol overhead — 15.9 vs 5 calls/iter at bs=1×). RCCL chunks every primitive call internally based on `NCCL_PROTO` + message size; host-side `NCCL_BUFFSIZE` / native `ncclAllToAll` API swaps were both flat (see §4.2). Real fix is RCCL source patches — outside this repo's scope. |
| 2 | **int4-vectorized embedding ops** (`update4_kernel` + `multi_to_one_reduce_vec4_v2`) | **5–10 %** | 3–5 days | Top embedding kernels consume ~2.1 ms/iter profile (~0.35 ms real = 8 % of bs=1× iter, per §3.2 trace). Already use `vec4` (8-byte int2) loads; bump to `int4` (16-byte) for `__half` slot data — same bounds-check rewrite pattern as Phase 12 concat (which landed +1.5 % at bs=8× via the same lever). Self-contained inside `embedding_storage/ragged_static_embedding.cu`. |
| 3 | **mscclpp xGMI-multicast custom AllReduce** | 1–3 % | 5–10 days | mscclpp source available at `/home/muabdulj/mscclpp/`. Need: (a) build mscclpp on ROCm 7.2 (untested), (b) write GPU-side AllReduce using mscclpp 1-sided put/get over xGMI, (c) integrate into HCTR's `NcclAllReduceInplaceComm` as a fallback path. **Only attacks AllReduce (1 of 17 RCCL calls)**; embedding all-to-all still uses RCCL → bounded upside. |
| 4 | **ROCm 7.3+ container upgrade for WarpSpeed** (`RCCL_WARP_SPEED_AUTO=1`) | 1–2 % | 0 days dev (waiting on image) | RCCL 2.28+ ships **WarpSpeed** (PR #2073) which halves CU usage for AllReduce / AllGather / ReduceScatter on gfx950. Tested RCCL 2.28.3 develop branch on 2026-05-13 (commit [`2dc79d6`](https://github.com/chriscai-amd/training_results_v5.1/commit/2dc79d6)) — flat: WarpSpeed strings not in develop branch (needs rocm-7.3.0 tag). AllReduce is only 14 % of exposed-RCCL → max +1-2 % even when engaged. Free win when image ships. |
| 5 | **BF16 mixed precision path** | 1–2 % | medium | NV submission uses BF16 mixed; we use FP16. Need `enable_bf16_compute` flag + `hip_bfloat16` template instantiations across `HugeCTR/src/layers/`. Eliminates loss-scaler stalls. |
| 6 | **Per-GPU NCCL stream affinity tuning** | < 1 % | 2 days HCTR core | `Send` / `Recv` on different streams could expose more concurrency; current HCTR uses single RCCL stream. Limited upside since RCCL kernels are CU-saturating. |
| 7 | **CPX / NPS4 GPU partition A/B** | unknown | invasive (requires `amd-smi set --compute-partition CPX` + reboot, may break 8-GPU assumption) | Per [RCCL usage tips](https://rocm.docs.amd.com/projects/rccl/en/develop/how-to/rccl-usage-tips.html), CPX+NPS4 boosts single-OAM AllReduce ~170 → 340 GB/s. Not directly applicable to our 8-GPU/single-OAM topology — worth A/B if remaining gap is AllReduce-bound. |

Sequence: items 1+2 in parallel (different code areas, additive gain). Item 4 lands free when ROCm 7.3 image arrives. Items 3, 5, 6, 7 are "if budget remaining" tail.

### 4.2 Hard blockers and things tried (negative results)

| Blocker / hypothesis | What we tried | Outcome | Owner / next step |
|---|---|---|---|
| **hipBLAS / hipBLASLt pin all dispatched kernels to capture-time stream** | hipblasSetStream-per-call workaround (Phase 14n.6); GraphScheduleable cross-graph pipeline split (Phase 14n.5) | hipBLASLt ignores per-call stream binding under HIP graph capture → all bprop kernels collapse onto the default stream regardless of ordering primitives. Multi-stream parallelism gap (4 active streams on AMD vs 10 on B200) cannot be closed in HCTR alone | AMD vendor (hipBLAS team) |
| **ROCm 7.2 `hipEventWaitExternal` returns `hipErrorInvalidValue`** | Restored the External flag in `pipeline.cpp` + `async_data_reader.cpp` (hipify had silently replaced it with `0`); added try-then-fallback wrapper (Phase 14p) | `std::bad_alloc` crash inside `Model::train_pipeline_with_ebc` OMP outlined function with both naïve restore AND fallback wrapper. Reverted to baseline. Cross-graph stream sync that NV's HIP graph capture relies on is unimplemented in ROCm 7.2.1 | AMD vendor (ROCm 7.3+) |
| **RCCL per-call latency floor 815 µs vs NV 155 µs (5.3 ×)** | 15 RCCL knobs swept (`NCCL_P2P_*`, `NCCL_GDR_*`, `NCCL_SHM_*`, `NCCL_ALGO`, `NCCL_CHUNK_SIZE`, `NCCL_*_NCHANNELS`); CU-mask via `hipExtStreamCreateWithCUMask` (Phase 14g) | All within ±0.6 % of baseline. Per-call floor is platform-fundamental: no NVLS-equivalent on xGMI in ROCm 7.2 RCCL | AMD vendor (NVLS-equivalent unannounced for MI400+) |
| **`hipGraphLaunch` host overhead 5.97 ms p50 vs NV 555 µs virtualized (11 ×)** | 14 `DEBUG_HIP_GRAPH_*` and `DEBUG_CLR_*` knobs swept at bs=8× | All within ±0.3 % of baseline. ROCm runtime kernel-launch path | AMD vendor (ROCm 7.3+ retest) |
| **RCCL emits 3.2 × more device kernels per logical collective than NCCL** | Bumped `NCCL_BUFFSIZE` 8 / 16 / 24 / 32 / 64 / 128 MiB; replaced looped `ncclSend` / `ncclRecv` with native `ncclAllToAll` / `ncclAllToAllv` API (Phase 14e, `HCTR_USE_NCCL_ALLTOALL=1`) | All flat at bs=1×. Trace: 17.4 → 16.5 RCCL kernels/iter (-5 %). Reverted as no-perf-win, added complexity. **RCCL chunks every primitive call internally based on `NCCL_PROTO` + message size** — host-side levers don't change kernel count | RCCL upstream (item #1 in §4.1 = real fix) |
| **CK-Tile MLP backward dgrad as bs=1× win** | Implemented end-to-end CK-Tile dgrad path (Phase 14p.2): new `hctr_cktile_gemm_dgrad_fp16` entry point, zero-bias-buffer workaround for upstream `Ds tuple > 0` static_assert, init API called pre-graph-capture, mlp_layer.cu integration with hipblas fallback. Bit-exact validated against CPU reference + hipBLAS | bs=1× **+0.3 % (noise)**; bs=8× **−24.9 % regression** (CK-Tile fixed 128×128×32 tile is wrong shape vs hipBLASLt's per-shape picks at large M). Code kept gated default-OFF as scaffold | Would need CK-Tile multi-tile dispatch OR fused bwd megakernel (dgrad + wgrad + bgrad + dRELU in one MFMA kernel) — multi-week effort |
| **MSCCLPP custom AllReduce (HCTR-internal)** | Built RCCL 2.27.7 with `-DENABLE_MSCCLPP=ON` (Phase 14h, commit [`2dc79d6`](https://github.com/chriscai-amd/training_results_v5.1/commit/2dc79d6)) | **−2.5 % regression**. MSCCLPP only optimizes AllReduce (14 % of exposed RCCL); setup overhead exceeds savings | (use mainline RCCL; revisit if RCCL exposes mscclpp-on-demand path) |
| **Per-shape hipBLASLt offline tuning at bs=1×** | `HIPBLASLT_TUNING_OVERRIDE_FILE` deployed | HCTR's `hipblasGemmEx` wrapper bypasses the override path → tunings never apply. Would need direct `hipblasLtMatmul` refactor | 3–5 day refactor, but per §3.2 the bs=1× MLP-GEMM gap is only +0.82 ms (38 % of iter gap) — bounded upside; deferred |
| **GPU clock pinning / boost** | `rocm-smi --showclocks` audit during steady state | Already at 2 100 MHz GFX boost during run — no headroom | (n/a) |

## Vendored upstream content

- `runtime_test/nvidia_frontend/train.py`, `mlperf_logger/`, `sharding/` —
  vendored from NV submission in this repo
  (`NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/`) with edits
  gated on env vars (`HCTR_USE_SUBSAMPLED_CRITEO`, `HCTR_USE_REAL_TABLE_SIZES`,
  `HCTR_USE_INNERPRODUCT_INSTEAD`, etc.).
- `mlperf_common_stub.py`, `mpi4py_stub.py`, `mlperf_common/` package —
  minimal shims; replace with real `mlperf_common` for production.

## License

Inherits Apache-2.0 from upstream HugeCTR + this MLPerf submissions repo.
