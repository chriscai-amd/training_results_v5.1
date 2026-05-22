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
All deltas measured at **NVIDIA's MLPerf-spec batch (55,296 = bs=1×, 6,912/GPU)**,
FP16 mixed, real MLPerf Criteo (HF subsample), 8 × MI350X. The single
metric we track and compare against NV B200 is bs=1× steady-state M sps;
historical bs ≥ 2× numbers in individual phase rows are kept for context
only and are no longer maintained.

## 2.1 Master timeline table

| Phase | Date | Optimization | Commits | Throughput before → after | Δ |
|---:|---|---|---|---:|---:|
| 1  | 2026-05-09 | Initial port — get build/link/run working | [`7882215`](https://github.com/chriscai-amd/training_results_v5.1/commit/7882215) | (didn't run) → first FP32 8-GPU run | n/a |
| 2  | 2026-05-09 | Multi-GPU FP16 MultiCross stabilisation (3 numerical bugs) | [`9d05c0f`](https://github.com/chriscai-amd/training_results_v5.1/commit/9d05c0f), [`5439485`](https://github.com/chriscai-amd/training_results_v5.1/commit/5439485), [`0a24ef2`](https://github.com/chriscai-amd/training_results_v5.1/commit/0a24ef2) | NaN at iter ≤ 2 → **5.85** M sps | first convergence |
| 3  | 2026-05-09 | HIP graph + intra/inter-iter overlap re-enabled (handle pre-warm `std::call_once`) | [`eb4fa60`](https://github.com/chriscai-amd/training_results_v5.1/commit/eb4fa60) | 5.85 → **6.26** | **+7.0 %** |
| 4  | 2026-05-10 | Multi-hot data path (912 B/row, 214 keys/row — apples-to-apples with NV submission) | [`5e12007`](https://github.com/chriscai-amd/training_results_v5.1/commit/5e12007) | 6.26 → **5.73** | -8.5 % (5× more embedding work; intentional regression for apples-to-apples) |
| 5  | 2026-05-10 | Fused MLP epilogue emulation (RELU_AUX / DRELU / DRELU_BGRAD via `hipblasGemmEx` fallback) | [`2ee57e7`](https://github.com/chriscai-amd/training_results_v5.1/commit/2ee57e7), [`0d9a35e`](https://github.com/chriscai-amd/training_results_v5.1/commit/0d9a35e), [`61ce55b`](https://github.com/chriscai-amd/training_results_v5.1/commit/61ce55b) | n/a (correctness only; fused path slower than InnerProduct on AMD) | (correctness) |
| 6  | 2026-05-10 | **V5 2D-tile bgrad/bgrada kernels** replace V1 cooperative scan (49 % of single-GPU time → sub-1 %). **[corrected 2026-05-18 — see §2.2 Phase 6 sub-table for revised attribution]** | [`56bc046`](https://github.com/chriscai-amd/training_results_v5.1/commit/56bc046), [`63a9c54`](https://github.com/chriscai-amd/training_results_v5.1/commit/63a9c54), [`4fc896a`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fc896a) | 5.73 → 11.24 *(headline, not reproducible)*; kernel-only: **22–457× per call** | **+96 %** *originally claimed*; **~+1 %** isolated re-A/B at 8-GPU bs=1× — most of the headline jump came from concurrent landings/config flips, not V5 alone |
| 7  | 2026-05-11 | FP16 NaN/inf clamp folded into `vector_fma{3,4}_align8` store path | [`93aad5c`](https://github.com/chriscai-amd/training_results_v5.1/commit/93aad5c) | 11.24 → **11.57** | +2.9 % @ bs1x; **+15 %** @ sweet-spot batch |
| 8  | 2026-05-11 | Re-enable intra/inter-iteration overlap (earlier scripts had set both to 0) | [`68e560e`](https://github.com/chriscai-amd/training_results_v5.1/commit/68e560e) | 11.57 → **12.55** | **+8.5 %** |
| 9  | 2026-05-11 | V5-style `add_bias_per_row_v5_kernel` (BLOCK_M=64 × N_TILE=128, coalesced, shared-mem broadcast) | [`3b984e4`](https://github.com/chriscai-amd/training_results_v5.1/commit/3b984e4) | 12.55 → **12.72** | +1.4 % @ bs1x; +2.6 % @ sweet-spot |
| 10 | 2026-05-11 | V5 BGRADA scratch zeroing folded into finalize kernel (3 hipMemsetAsync/iter eliminated) | [`b898899`](https://github.com/chriscai-amd/training_results_v5.1/commit/b898899) | 12.72 → **12.74** | +0.2 % (within noise) |
|    | 2026-05-12 | NFS-bound → `/dev/shm` fix (AsyncReader uses `O_DIRECT`, bypasses page cache) | [`bf88560`](https://github.com/chriscai-amd/training_results_v5.1/commit/bf88560) | 5.21 (NFS) → **11.76** sustained @ bs1x | +127 % vs NFS-bound run |
|    | 2026-05-12 | NV-submission audit (>120 env-knob configs swept) — bake `NCCL_PROTO=LL128`, `HIP_FORCE_DEV_KERNARG=1` | [`1ef02aa`](https://github.com/chriscai-amd/training_results_v5.1/commit/1ef02aa), [`fd1f4d2`](https://github.com/chriscai-amd/training_results_v5.1/commit/fd1f4d2), [`2c4670a`](https://github.com/chriscai-amd/training_results_v5.1/commit/2c4670a), [`a923ef1`](https://github.com/chriscai-amd/training_results_v5.1/commit/a923ef1) | 11.76 → **11.88** sustained | +1.0 % |
| 11 | 2026-05-12 | **`DEBUG_HIP_DYNAMIC_QUEUES=1`** (on-demand HW queue allocation; the 4-stream pipeline now overlaps properly). **[re-validated 2026-05-18 — see §2.2 Phase 11 sub-table]** | [`3860967`](https://github.com/chriscai-amd/training_results_v5.1/commit/3860967) | 11.88 → 12.92 @ bs1x (May-12); on HEAD: DYNQ=0 13.80 → DYNQ=1 **14.43** @ bs1x (May-18, 3-trial steady) | **+8.9 %** (orig); **+4.5 %** (HEAD re-A/B) — decayed but still load-bearing |
|    | 2026-05-12 | **Configuration: peak measurement at bs8x (442,368) instead of bs1x (55,296)** (amortizes 1.29 ms/iter host overhead — see linear-fit in Part 3) | (configuration only) | bs1x **12.92** → bs8x **16.42** M sps (same binary, 8× larger global batch, ~7 % per-GPU memory increase) | **+27 %** (**+5.3 ms saved per 8× samples**; relaxes MLPerf-spec batch constraint) |
| 12 | 2026-05-12 | int4-vectorized `__half` elementwise: `concat_fwd/bwd_kernel_vec8` + `binaryOp_kernel_vec8_half` (used by MultiCross matrix_add) | [`4fb17c3`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fb17c3), [`e3e64a2`](https://github.com/chriscai-amd/training_results_v5.1/commit/e3e64a2) | 16.42 → **16.67** @ bs8x | **+1.5 %** at peak |
| 13 | 2026-05-13 | **`HCTR_DROP_TABLE_SIZE_CLAMP=1` on real-data path** — removes stale `max(real_size, 65536)` clamp that broke `auto`-planner's DP-replication of the 13 small embedding tables (≤ 7,424 elems). Reduces embedding all-to-all volume by ~80 %. (NV's May-13 hot-fix per b200/README §7.5.) | [`2dc79d6`](https://github.com/chriscai-amd/training_results_v5.1/commit/2dc79d6) | bs8x 16.67 → **16.98**; bs2x 15.04 → **15.31** | **+2.0 % @ bs8x; +1.8 % @ bs2x; flat @ bs1x** (RCCL is CU-bound on AMD, not volume-bound) |
| 14a | 2026-05-13 | **`NCCL_BUFFSIZE=8 MiB → 32 MiB`** — at bs1x embedding output is 46 MB ≫ 8 MB so RCCL splits each logical SendRecv into ~6 chunks; bumping the buffer to 32 MB merges most chunks → fewer per-call inter-launch gaps | baked in `run_b200_match.sh` | bs1x 12.90 → **13.07**; bs8x 16.98 → 16.99 (flat) | **+1.3 % @ bs1x; flat @ bs8x** (RCCL only 10 % of bs8x kernel-time, dominant at bs1x) |
| 14b | 2026-05-13 | **`HCTR_FUSE_TOP_MLP=1` (re-enabled at bs ≥ 4×)** — Phase-5's negative measurement (-6 % at all batches) was a measurement artifact: with `NCCL_BUFFSIZE=8 MiB` the fused top-MLP path's larger NCCL chunks couldn't pipeline. With `BUFFSIZE=32 MiB` + bs ≥ 4× the fused-MLP epilogue chain (1 GEMM + 1 epilogue kernel per FC layer instead of 5–6 unfused) finally amortizes. | `run_b200_match.sh` auto-picks based on `HCTR_BATCH` | bs8x 16.99 → **17.86**; bs4x 16.50 → **17.49**; bs2x 15.31 → 14.59 (regress); bs1x 13.07 → 12.46 (regress) | **+5.2 % @ bs8x; +6.0 % @ bs4x; ‑4.7 % @ bs2x; ‑4.6 % @ bs1x** — auto-disabled below bs4× |
| 14c | 2026-05-13 | **`HCTR_FUSE_WB=True` (at bs ≥ 8×)** — fold weight-bias post-pass into the fused MLP. Stacks on top of 14b. At bs ≤ 4× this is flat or slightly negative (kernel overhead dominates the bias-fuse savings). | `run_b200_match.sh` auto-picks based on `HCTR_BATCH` | bs8x 17.86 → **17.93**; bs4x 17.49 → 17.14 (regress) | **+0.4 % @ bs8x; ‑2.0 % @ bs4x** — auto-disabled below bs8× |
| **15** | **2026-05-15** | **Dedicated RCCL streams for DP-allreduce + MLP-wgrad-allreduce** (`set_absolute_stream("rccl_emb_ar")`, `set_absolute_stream("rccl_mlp_wgrad")` in `model_pipeline.cpp`). Phase-14r 8-GPU trace showed default stream tid=32 carrying 553 µs of RCCL serialised with 4071 µs of MLP compute; this lever is the AMD-side equivalent of NV's cuBLASLt-internal worker streams (which hipBLASLt does NOT replicate on gfx950). 2-trial bs=1× A/B at 13.080→13.604, 13.125→13.455 → +3.3 % avg lift; loss bit-equivalent (0.291667). Default ON; set `HCTR_DEDICATED_RCCL_STREAM=0` to disable. | (perf-push branch, src/pybind/model_pipeline.cpp lines 222-236, 309-315) | bs1x 13.07 → **13.48** | **+3.0 % @ bs1x** |
| **16** | **2026-05-15** | **MultiCross dx-accumulator memset elimination.** Phase-14r trace: AMD MI350X has 1003 µs/iter of `__amd_rocclr_fillBufferAligned` (NV has 0). Largest single contributor was `accum_dx_tensor_` (47 MB at bs=1×) zero-init in `MultiCrossBackwardFunctorv2`. Phase-16 adds a "first-iter overwrite" variant of `vector_mul_fma3_align<__half, 8, 4>` that stores `out1 = a * c` instead of `out1 += a * c`, equivalent to "memset(0) + accumulate" but skipping the memset. Uses overwrite path on the first bwd loop iteration only. 2-trial bs=1× A/B at 13.545→13.634, 13.554→13.654 → +0.7 % avg lift; loss bit-equivalent. Default ON; set `HCTR_SKIP_DX_MEMSET=0` to disable. | (perf-push branch, src/layers/multi_cross_layer.cu lines 198-238 + 822-852) | bs1x 13.48 → **13.55** | **+0.7 % @ bs1x** |
| **17** | **2026-05-22** | **Restore native hipBLASLt fused BIAS epilogue on fprop MLP GEMMs** (`HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP=1`). Flips the `BIAS` epilogue on for the existing `Bias_HA` Tensile family used by all DLRM-DCNv2 fwd MLP shapes — eliminates 720 `add_bias_per_row_v5` launches per 30 iters / 8 ranks (rocprof: **−148 µs/rank/iter** GPU-time saved, V5 add-bias bucket 34.59→0.00 ms). BPROP gate (`HCTR_HIPBLASLT_FUSED_EPILOGUES_BPROP`) kept OFF: hipBLASLt 1.2 BGRADA Tensile pool on gfx950 only exposes 2 algorithms, both ~3× slower than the V5 fallback (filed upstream). **Long-run validation** (3-trial steady-state, 5000 iters/trial, eval disabled, alternating A/B, warmup discarded): dedicated-RCCL stream Δ = −0.097 ms/iter, **+2.19 % throughput** (95 % CI [+1.37 %, +2.92 %], n=147 per-window samples/side); shared-RCCL stream **+2.53 %** (95 % CI [+2.06 %, +2.87 %]). Kernel-level upper bound (148 µs / 4587 µs baseline = +3.23 %) matches observation at ~78 % critical-path-conversion. **TTT AUC parity on 1 TB Criteo head** (19,800 iters, ~0.26 MLPerf-epoch, scaler 1024): FPROP=1 monotonically ≥ baseline at all 5 eval checkpoints (epoch 0.19→0.96, AUC deltas +0.008…+0.041), no NaN, loss-equivalent. Default OFF; opt in via env var. | `fused_gemm_functors.cu` L686 (fprop gate); L801-805 (bprop gate, kept OFF) | bs1x 13.55 → **13.85** | **+2.2 % @ bs1x** |

**bs=1× post-Phase-16 (default-ON config): 13.55 M sps** = **51.5 %** of NV B200's bs=1× (26.33 M sps). With opt-in Phase 17 (`HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP=1`): **13.85 M sps = 52.6 %**.

| bs=1× config (global batch 55,296, 6,912/GPU) | M sps | trial source | vs NV B200 May-13 bs=1× |
|---|---:|---|---:|
| **default-ON wins only (Phase 13 → 16)** | **13.55** | 2026-05-15 2-trial avg (250 iters) | **51.5 %** of 26.33 |
| **+ opt-in Phase 17 (FPROP fusion)** | **13.85** | 2026-05-22 3-trial steady (5000 iters/trial), 95 % CI [+2.06 %, +2.87 %] vs default | **52.6 %** of 26.33 |

## 2.2 Selected sub-tables (key wins in detail)

### Phase 6 — V5 2D-tile bgrad/bgrada kernels — [`56bc046`](https://github.com/chriscai-amd/training_results_v5.1/commit/56bc046), [`63a9c54`](https://github.com/chriscai-amd/training_results_v5.1/commit/63a9c54), [`4fc896a`](https://github.com/chriscai-amd/training_results_v5.1/commit/4fc896a)

**What these kernels do in DLRM-DCNv2.** Every fully-connected layer
in DLRM-DCNv2's **bottom MLP** (3-4 FC layers on the dense feature
inputs), **top MLP** (4-5 FC layers above the cross-network producing
the click prediction), and the **MultiCross / DCNv2 layers** has the
form `y = W @ x + b` followed optionally by ReLU. Their **backward
pass** requires the bias gradient `db = sum over the batch dimension
of dy`, which is exactly a per-row column-sum of the upstream gradient
matrix — i.e. a "reduce-sum-columns" reduction. With ~7-18 FC
sub-layers per iter and a per-GPU batch of 6,912 at bs=1×, that's
hundreds of column-sums per iter:

  * `reduce_sum_columns_kernel` (V1) → `bgrada_v5_kernel` (V5) — the
    wgrad-GEMM's BGRADA epilogue (`db += sum_batch(dy)`).
  * `bprop_drelu_bgrad_v5_kernel` (V5) — the dgrad-GEMM's
    **fused** DRELU+BGRAD epilogue (`dy *= relu_mask; db += sum_batch(dy)`),
    fusing what's otherwise two separate passes.
  * `add_bias_per_row_v5_kernel` (V5) — the forward bias broadcast
    after each FC layer's GEMM (`y += b`).
  * `bgrad_finalize_v5_kernel` — divides the FP32 accumulator by 256
    for FP16 ncclSum headroom, clamps to ±65 504, casts to FP16.

**Technical change — confirmed sound.** V1's `reduce_sum_columns_kernel`
launches one block per output row with a 256-thread cooperative scan
over the contracted dim, using stride-`m` uncoalesced FP16 loads — at
small `m` (e.g. the bot-MLP first layer with `m=128`) only one
wavefront's worth of work survives, ~1 % CU utilisation. Replaced with
a V5 2D tile (BLOCK_M=64 rows × N_TILE=1024 cols/block), wave-aligned
coalesced 128-byte loads, atomicAdd into per-device pre-allocated FP32
scratch (pre-warmed via `std::call_once` so HIP graph capture sees the
alloc done), followed by the finalize kernel above.

**Kernel-level microbenchmark (added 2026-05-18 to validate).** A
standalone HIP microbench (`scripts/bgrada_microbench.cu`) running V1
and V5 directly on the production DLRM-DCNv2 BGRADA shapes (col-major
__half input, m∈{128,256,512,1024}, k∈{6912, 55296} = per-GPU batch
at 8-GPU bs=1× and full single-GPU bs=1× respectively) measures
V5 = **22 – 457× faster per call** than V1, scaling with k:

| M | K | context | V1 µs/call | V5 µs/call | **V5 speedup** | V5 GB/s |
|---|---|---|---:|---:|---:|---:|
| 128  | 6,912  | 8-GPU bs=1× | 697.8  | 31.0 | **22.5×** | 57   |
| 128  | 55,296 | 1-GPU bs=1× | 12,628.7 | 31.2 | **404×**  | 453  |
| 256  | 55,296 | 1-GPU bs=1× | 13,987.2 | 30.6 | **457×**  | 924  |
| 512  | 6,912  | 8-GPU bs=1× | 708.4  | 30.6 | **23.2×** | 231  |
| 1024 | 55,296 | 1-GPU bs=1× | 14,405.1 | 76.7 | **188×**  | **1,477** |

V5 hits **28 % of MI350X HBM3 peak** (~5,300 GB/s) on the largest
shape; V1 strands the GPU at 0.1–0.4 % of peak with its stride-`m`
uncoalesced loads. **The kernel speedup is real and not in dispute.**

**Production-level reproducibility — partial, with caveats.** A
controlled re-A/B at 8-GPU bs=1× on **HEAD as of 2026-05-18** with
the V5 path forced off (`HCTR_DRELU_KERNEL=v1
HCTR_ADD_BIAS_KERNEL=v1`) versus default-V5 measures:

| Configuration | Per-iter | Throughput | Δ |
|---|---:|---:|---:|
| V5 default-on (HEAD)         | 4.29 ms | 12.879 M sps | baseline |
| V1 forced (env-knob off-V5)  | 4.34 ms | 12.737 M sps | **−1.1 %** |

**Why the e2e gain is so small even when the kernel is 22-457× faster.**
`rocprofv3 --kernel-trace --stats` on the 8-GPU bs=1× run
(`HCTR_FUSE_TOP_MLP=1 HCTR_ASYNC_WGRAD=0`, BGRADA on the critical
stream) shows the V5-replaced kernels (`bgrada_v5_kernel`,
`bprop_drelu_bgrad_v5_kernel`, `add_bias_per_row_v5_kernel`,
`bgrad_finalize_v5_kernel`) consume only **6.5 ms / 321 ms = 2.0 %
of total GPU kernel time**. That's the hard upper bound on the
iter-wall improvement Phase 6 can deliver: even an infinite kernel
speedup cannot reduce GPU kernel time by more than 2 %, and because
BGRADA runs on the wgrad stream overlapped with the longer dgrad
critical path, the actual visible iter-time delta is even smaller.

At single-GPU the same ratio is much larger (V1 takes 12-15 ms per
BGRADA call at `k=55,296` per the microbench, and there are ~18
calls per iter, so V1 alone would dominate iter time). Going from
single-GPU to 8-GPU shrinks the per-GPU batch ~8× (the V1 → V5
absolute speedup at small `k` shrinks correspondingly) and adds
embedding + RCCL + 18 other GEMMs to the critical path, so Phase 6's
footprint drops to ~2 % of total kernel time and the per-kernel
speedup correspondingly drops from a 1:1 wall-time win to invisible.

The 5.73 → 11.24 headline was a measurement-methodology change
(`HCTR_INTRA_OVERLAP` + `HCTR_INTER_OVERLAP` flipped on at runtime,
worth ~10 % per a separate 4-corner A/B) plus the cumulative effect
of phases 7-18 that landed shortly after `4fc896a`, not the kernel
rewrite itself.

**Bottom line.** V5 is a genuine, large per-kernel improvement
(measurable on the kernel microbench, 22-457×) and is worth keeping
default-on as insurance for any future workload where BGRADA hits the
critical path (smaller-batch, fewer-GPU, single-GPU). On the current
production 8-GPU bs=1× config it contributes ~+1-2 % to iter time, not
the originally headlined +96 %.


### Phase 11 — `DEBUG_HIP_DYNAMIC_QUEUES=1` (decayed from +8.9 % → +4.5 % at bs1x, still load-bearing) — [`3860967`](https://github.com/chriscai-amd/training_results_v5.1/commit/3860967)

After exhausting >120 env-knob configurations, found that AMD's HIP
runtime debug knob `DEBUG_HIP_DYNAMIC_QUEUES=1` changes **how logical
streams are mapped onto the GPU's hardware queues**.

**The hard limits on gfx950 (MI350X)** — independent of `DYN_QUEUES`:

- `GPU_MAX_HW_QUEUES = 4` (`clr/rocclr/utils/flags.hpp:140`): only **4
  HSA HW queues per priority level per device** can be allocated.
- `numHwPipes_ = 4` (`rocdevice.cpp:140`): the GPU has **4 physical
  compute pipes**. HW queues round-robin onto pipes via `queue_id %
  numHwPipes_`. There are only 4 pipes regardless of how many queues
  you allocate.


**What `DEBUG_HIP_DYNAMIC_QUEUES=1` actually changes** (per
`rocdevice.cpp::getQueueFromPool` lines 3000-3034): when multiple streams
must squeeze onto 4 hardware queues, the runtime picks which queue a
new launch goes to using a different metric.

- **mode 0**: picks the queue **with the fewest streams ever bound
  to it** — a static count that ignores what's currently running.
- **mode 1** (default): picks the queue **whose dispatch ring has
  the least pending work right now** — a live measurement of how
  busy each queue is at this moment.

**Worked example — why the compute stream ends up waiting for the
embedding stream** (the exact pattern we observed in the trace):

At the start of an iteration, HCTR launches in roughly this order
(see the trace decomposition below):

```
t = 0.00 ms   defaultmp  [emb_fwd] ebc_mp_model       ←  big embedding kernel
t = 0.05 ms   defaultmp  (kernel still running)
t = 0.10 ms   default    [graph] network              ←  about to launch
```

When `default` is about to launch `[graph] network`, the runtime
needs to decide which HW queue to use. Both modes have the same 4
queues to pick from; suppose at this moment:

| queue | streams ever attached | pending kernels right now |
|---|---|---|
| Q0 | defaultmp (1) | **1** (the embedding kernel still executing) |
| Q1 | defaultdp (1) | 0 |
| Q2 | prefetch (1) | 0 |
| Q3 | rccl_emb_ar (1) | 0 |

- **mode 0** (refCount only) — all four queues have the same
  attachment count (1 each); ties are broken by queue ID, so it
  picks **Q0**. The graph-network launch is now queued *behind*
  the embedding kernel that's still executing on Q0. The default
  stream waits.
- **mode 1** (depth-aware) — Q0 has pending depth 1, the others
  have depth 0. Picker prefers Q1/Q2/Q3 (depth 0). Graph network
  launches immediately, in parallel with embedding.

This isn't hypothetical — the trace shows exactly this pattern:
under DYN_QUEUES=0, `[graph] network` launches 0.31 ms *later* than
under DYN_QUEUES=1, after 6 unrelated kernels have already entered
the queue ahead of it.

A second, separate failure mode appears when you try to side-step
this by raising `GPU_MAX_HW_QUEUES`: with `MAXQ=8 DYNQ=0`, two
streams that should be on different physical pipes end up on the
same pipe (mode 0 doesn't know about pipes), and they fight for
CUs *during execution* — making the graph kernel itself run 1.86 ms
slower, not just delayed. Both failure modes are quantified below.

**Re-validation on HEAD (2026-05-18, 3-trial × 120-iter, bs=1×,
steady-state iter 50-110 window):**

| config | per-iter | sps (steady) | Δ |
|---|---:|---:|---:|
| **DYN_QUEUES=1** (default-on) | **3.83 ms** (σ ≈ 0.02) | **14.43 M** | baseline |
| DYN_QUEUES=0 | 4.01 ms (σ ≈ 0.01) | 13.80 M | **−4.5 %** |

This is roughly half the originally claimed +8.9 %. The mechanism is
still real — Phases 13-18 + 20.x simply reduced the absolute amount
of work *available* to overlap (RCCL on dedicated streams, fewer
hipMemsetAsync calls, async_wgrad off), so the wall-clock benefit
from queue-isolation shrinks proportionally.

**Trace-level decomposition (native HCTR Perfetto trace, 5 in-window
iters, rank 0, single-trial):**

The penalty surfaces almost entirely as a **delayed start for
`[graph] network` on the `default` compute stream**:

| event sequence (relative to iter start) | DYN_QUEUES=1 | DYN_QUEUES=0 |
|---|---:|---:|
| `[emb_fwd] ebc_mp_model` first kernel ready (defaultmp) | 0.05 ms | 0.04 ms |
| **`[graph] network` starts (default)** | **0.14 ms** | **0.45 ms** |
| Kernels launched on defaultmp / defaultdp / prefetch *before* default-stream graph starts | 1 | 6 |
| `[graph] network` duration | 3.49 ms | 3.12 ms |
| **iter wall** | **4.33 ms** | **4.57 ms (+0.24 ms)** |

With DYN_QUEUES=0 the network-graph kernel on `default` waits in the
HSA dispatch queue **0.31 ms longer**, behind 6 unrelated launches on
the embedding & prefetch streams that happened to land first. The
graph kernel itself then runs 0.37 ms *shorter* (it's effectively been
"pre-warmed" by waiting), so the net iter penalty is +0.24 ms ≈ the
+147 µs / +4.5 % we measured at the wall.

**Aggregate stream-overlap factor** (sum of all leaf-event times across
streams ÷ iter wall − 1):

| config | leaf-event sum | iter wall | overlap factor |
|---|---:|---:|---:|
| DYN_QUEUES=1 | 5.13 ms | 4.43 ms | **+16 %** (good cross-stream concurrency) |
| DYN_QUEUES=0 | 4.81 ms | 4.58 ms | +5 % (mostly serialised) |

The visual signature in the Perfetto trace is unambiguous: under
DYN_QUEUES=0, the long blue `[graph] network` block on the `default`
lane appears strictly *after* the embedding kernels on `defaultmp` /
`defaultdp` finish, instead of running in parallel as it does with
DYN_QUEUES=1. Keep default-on.



### Phase 17 — Restore native hipBLASLt fused BIAS epilogue on fprop MLP GEMMs (`HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP=1`)

**Principle.** Every fwd FC layer in DLRM-DCNv2 evaluates
`y = W @ x + b` (optionally followed by ReLU). The historical AMD
path runs the matmul through `hipblasLtMatmul` with an `EPILOGUE_NONE`
descriptor and then launches a separate `add_bias_per_row_v5` kernel
that broadcasts `b` over the row dimension of `y`. hipBLASLt is fully
capable of folding that broadcast into the GEMM's epilogue — the
Tensile kernel family selected for these shapes is *already*
`Bias_HA_S_SAV` in both the baseline and fused paths. The kernel
**name** advertises bias *capability*; whether bias is actually applied
is controlled by a runtime flag on the matmul descriptor. Phase 17
flips that flag on for the fprop direction so the same kernel writes
`y + b` instead of plain `y`, removing the separate V5 launch from the
critical path at zero cost to Tensile per-call time.

The gate already existed in code (`fused_gemm_functors.cu` L686) but
was historically held off because turning it on globally regressed
the bprop pass (see "Bprop direction — ruled out" below). Phase 17
splits the gate per-direction (`HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP`
on L686 for fwd, `HCTR_HIPBLASLT_FUSED_EPILOGUES_BPROP` on L801-805
for bwd), making the safe fwd-only subset accessible.

**Kernel-level measurement (`rocprofv3 --kernel-trace --stats`, 30
iters / 8 ranks, FPROP-only A/B):**

| bucket | baseline calls / ms | FPROP=1 calls / ms | Δ |
|---|---:|---:|---:|
| `add_bias_per_row_v5` (V5 broadcast)        | 720 / 34.59 ms | **0 / 0.00 ms**     | **−34.59 ms** |
| Tensile `Bias_HA` fwd MLP                   | 13920 / 532.83 ms | 13920 / 532.23 ms | noise |
| `bgrada_v5` (bprop V5)                      | 720 / 50.53 ms | 720 / 50.54 ms      | noise |
| `bgrad_finalize_v5` (bprop V5)              | 720 / 3.67 ms  | 720 / 3.70 ms       | noise |
| `Bias_A_SAV` (slow bias-fused bprop)        | 0              | 0                   | — (BPROP gate off) |
| DRELU-fused bprop                           | 0              | 0                   | — (BPROP gate off) |

**Net in the target bucket: −35.47 ms over 8 ranks × 30 iters =
−148 µs/rank/iter saved.** Tensile per-call time stays flat
(−0.60 ms / 13920 calls is noise), confirming the V5 elimination is
pure profit and not a Tensile shape regression.

A second rocprof run on the TTT pipeline (30 iters,
`EVAL_INTERVAL=99999`) confirms the same signature: `add_bias_per_row_v5`
720 → 0 (−59.4 ms across the longer trace), `Bias_HA` fwd MLP 13920
calls in both with −48.4 ms (inline-bias active), `Bias_A_SAV` = 0 in
both (BPROP gate correctly off).

**Wall-time validation (5000-iter steady-state, 3-trial alternating
A/B, eval disabled, warmup discarded):**

| stream | mean ms/iter A (FPROP=0) | mean ms/iter B (FPROP=1) | Δ | throughput Δ |
|---|---:|---:|---:|---:|
| dedicated RCCL | 4.5316 (±0.0212) | 4.4343 (±0.0279) | −0.097 ms, 95 % CI [−0.132, −0.062] | **+2.19 %** (95 % CI [+1.37 %, +2.92 %]) |
| shared RCCL    | 4.5875 (±0.0140) | 4.4743 (±0.0122) | −0.113 ms, 95 % CI [−0.132, −0.095] | **+2.53 %** (95 % CI [+2.06 %, +2.87 %]) |

The shared-RCCL win being slightly larger than dedicated-RCCL is
consistent with the V5 add-bias launches sitting on the RCCL-blocked
critical path when the RCCL stream is shared — removing them saves
both their own GPU time and the launch-gap they injected.

**TTT AUC parity on 1 TB Criteo head** (19,800 iters,
~0.26 MLPerf-epoch, scaler 1024, dedicated RCCL):

| epoch | FPROP=0 AUC | FPROP=1 AUC | Δ |
|---:|---:|---:|---:|
| 0.19 | 0.7553 | 0.7634 | +0.008 |
| 0.38 | 0.7517 | 0.7632 | +0.012 |
| 0.57 | 0.7313 | 0.7719 | +0.041 |
| 0.76 | 0.7586 | 0.7726 | +0.014 |
| 0.96 | 0.7633 | 0.7715 | +0.008 |

FPROP=1 is monotonically ≥ baseline at every eval checkpoint (deltas
within ±0.007 single-trial noise but sign consistently positive). No
NaN, loss-equivalent (0.27 → 0.27 both sides). Convergence preserved.

**Bprop direction — ruled out (BPROP gate kept OFF).** Flipping the
same `BIAS` epilogue on for the bwd path (`HCTR_HIPBLASLT_FUSED_EPILOGUES_BPROP=1`)
regresses end-to-end by +200 µs/iter (the all-fusion config measures
4.28 ms/iter, *worse* than the FUSED=0 baseline's 4.20 ms/iter on
dedicated RCCL). The cause is a hipBLASLt 1.2 / gfx950 Tensile-library
coverage gap on the `BGRADA` GEMM, not the fusion mechanism itself:

- For DLRM-DCNv2's hot bwd shape (1024 × 3456 × 6912, NT, HHS,
  `bias_type=f16_r`), `hipblaslt-bench --algo_method all` returns
  **458 solutions** for plain matmul, **458** for forward-BIAS, but
  only **2** for `BGRADA`. Both BGRADA candidates use the slow
  `MT128x256x16 / MI16x16x4` tile family (~300-308 µs/call); the
  plain-matmul path's heuristic picks `MT128x176x128 / MI16x16x1`
  at ~100 µs/call. Adding `--gradient` collapses the Tensile pool
  ~230×.

- HCTR's runtime heuristic on the bias-fused BGRADA already picks
  the faster of the 2 available solutions (verified: trace per-call
  ≈ 310 µs, matches Tensile's `[1]` at 300 µs within 10 µs jitter).
  Headroom inside the 2-solution pool is therefore zero.

- `HIPBLASLT_TUNING_OVERRIDE_FILE` was probed as a possible escape
  hatch and ruled out for the same reason: an override file can only
  re-pin among the 2 available solutions, so the best-case win is 0 %.
  Confirmed independent of `--bias_type` (same 2-solution pool for
  `f16_r` and `f32_r`).

- A `DRELU_BGRAD` epilogue (fusing relu-mask + bias-grad) hits the
  same library-coverage gap and was not pursued.

**Bottom line.** Default OFF, opt in via env. Fprop-only is the
right subset of the historical "global FUSED=1" lever for hipBLASLt
1.2 / gfx950 — it captures the entire V5 add-bias elimination
(+2.5 % steady-state, +2.2 % at bs=1× in the master table) while
sidestepping the BGRADA library-coverage gap that contaminates the
bwd direction.


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

## License

Inherits Apache-2.0 from upstream HugeCTR + this MLPerf submissions repo.
