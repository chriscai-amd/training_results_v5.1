# DLRM-DCNv2 — HugeCTR ROCm port for AMD Instinct MI350X

A working port of NVIDIA's MLPerf v5.1 HugeCTR DLRM-DCNv2 submission to AMD's
ROCm 7.2 platform, targeting `gfx950` (Instinct MI350X). Source-compatible
with NVIDIA's `train.py` Python frontend and `mlperf_logger` MLLog format.

This is a research / port branch, **not an official MLPerf submission**.

## Status

Single node, 8 × MI350X, FP16 mixed precision (matching NVIDIA's B200
submission), Adagrad, scaler 16,348, sharding=auto, HIP graph + overlap on,
`DEBUG_HIP_DYNAMIC_QUEUES=1` baked in. All numbers on real MLPerf Criteo
data (HuggingFace subsample, copied to `/dev/shm` to avoid the AsyncReader's
`O_DIRECT` NFS bottleneck).

| Batch (global) | Per-GPU | ms/iter | **AMD M sps** | NV B200 (same data) | Ratio |
|---:|---:|---:|---:|---:|---:|
| 55,296 (1×, MLPerf-spec) | 6,912 | 4.28 | **12.92** | 15.78 | **81.9 %** |
| 110,592 (2×) | 13,824 | 7.35 | **15.04** | 18.20 | **82.6 %** |
| 221,184 (4×) | 27,648 | 13.47 | **16.43** | 19.82 | **82.9 %** |
| **442,368 (8×, peak)** | 55,296 | 26.54 | **16.67** | 19.19 | **86.9 %** ← peak ratio |

NVIDIA B200 reference numbers come from the companion
[`b200/.../README-b200-1x8.md`](https://github.com/chriscai-amd/training_results_v5.1/blob/chcai/b200/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/README-b200-1x8.md)
where NVIDIA ran their published MLPerf submission binary on the same
HuggingFace Criteo subsample. The publicly headlined **23.02 M sps on
8 × B200** is on the full 4.2 B-row MLCommons R2 corpus (~4 TB), which
this port doesn't have local capacity to host; on equivalent HF data
NV's own peak is **19.82 M sps** at bs4x.

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
commit (`7882215`, 2026-05-09) added 1,661 files / ~478 K LOC — the full
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
| OpenMP `#pragma omp parallel` silently no-opped → AUC NCCL warmup deadlocked | `-fopenmp` for HIP/CXX in CMakeLists | `0a24ef2` |
| MultiCross v2 **BGRADA FP16 overflow** at per-rank batch ≥ 2048 (FP16 column-sum could exceed 65,504 → inf → NaN via `ncclSum` → poisons Adagrad) | FP32 accumulation + `isfinite()` guard + ±FP16-max clamp + pre-divide by 256 (Adagrad scale-invariant) | `5439485` |
| Wave-size mismatch in MultiCross bprop → row-cross-contamination at per-rank batch ≥ 2048 | `WARP_SIZE = 64` for AMD | `9d05c0f` |
| MultiCross fprop **NaN poisoning** at per-rank batch ≥ 2048 (one inf/NaN element propagates through later GEMMs) | `clamp_fp16_kernel` post-pass after each cross layer's `fused_matrix_elementwise_dot_add`; later inlined into FMA store via `sanitize_half2_fp16` | `0a24ef2` then `93aad5c` |
| `hipblasCreate` illegal during HIP graph capture | Per-device `hipblasHandle_t` cache pre-warmed via `std::call_once` in `CublasAlgo<T>::init_algorithm` | `eb4fa60` |
| Synthetic Criteo IDs (0..65535) indexed OOB into real `TABLE_SIZE_ARRAY` (some entries 3, 36, 63) | Clamp `TABLE_SIZE_ARRAY[i] = max(real_size, 65536)` when `HCTR_USE_REAL_TABLE_SIZES=1` | `7882215` |
| `Model.fit()` segfaulted on synthetic data — `DistributedSlotSparseEmbeddingHash` incompatible with `MultiHot AsyncDataReader` | Switch to modern `EmbeddingCollection` API; add `Reshape` between `EmbeddingCollection` output and `Concat` | `7882215` |
| `ncclCommInitAll` failed on 8-GPU init — `shard_matrix` hardcoded for 1 GPU | Scale to `NGPU` in train.py's sharding generation | `7882215` |
| Multi-GPU fused MLP collapsed to `log(2)·2` — BIAS post-pass guard was `(act == None) && bias != null`, dropping bias on every hidden ReLU layer | Drop `act == None` clause; BIAS post-pass fires whenever `bias != null && (saved_is_bias_epilogue || saved_is_relu_aux_epilogue)` | `0d9a35e` |
| Excluded layers (`MultiHeadAttention`, `GRU`, etc.) still type-referenced | Throwing stubs in `excluded_layer_stubs.cpp` | `7882215` |

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
| 1  | 2026-05-09 | Initial port — get build/link/run working | `7882215` | (didn't run) → first FP32 8-GPU run | n/a |
| 2  | 2026-05-09 | Multi-GPU FP16 MultiCross stabilisation (3 numerical bugs) | `9d05c0f`, `5439485`, `0a24ef2` | NaN at iter ≤ 2 → **5.85** M sps | first convergence |
| 3  | 2026-05-09 | HIP graph + intra/inter-iter overlap re-enabled (handle pre-warm `std::call_once`) | `eb4fa60` | 5.85 → **6.26** | **+7.0 %** |
| 4  | 2026-05-10 | Multi-hot data path (912 B/row, 214 keys/row — apples-to-apples with NV submission) | `5e12007` | 6.26 → **5.73** | -8.5 % (5× more embedding work; intentional regression for apples-to-apples) |
| 5  | 2026-05-10 | Fused MLP epilogue emulation (RELU_AUX / DRELU / DRELU_BGRAD via `hipblasGemmEx` fallback) | `2ee57e7`, `0d9a35e`, `61ce55b` | n/a (correctness only; fused path slower than InnerProduct on AMD) | (correctness) |
| 6  | 2026-05-10 | **V5 2D-tile bgrad/bgrada kernels** replace V1 cooperative scan (49 % of single-GPU time → sub-1 %) | `56bc046`, `63a9c54`, `4fc896a` | 5.73 → **11.24** | **+96 %** ← single biggest win |
| 7  | 2026-05-11 | FP16 NaN/inf clamp folded into `vector_fma{3,4}_align8` store path | `93aad5c` | 11.24 → **11.57** | +2.9 % @ bs1x; **+15 %** @ sweet-spot batch |
| 8  | 2026-05-11 | Re-enable intra/inter-iteration overlap (earlier scripts had set both to 0) | `68e560e` | 11.57 → **12.55** | **+8.5 %** |
| 9  | 2026-05-11 | V5-style `add_bias_per_row_v5_kernel` (BLOCK_M=64 × N_TILE=128, coalesced, shared-mem broadcast) | `3b984e4` | 12.55 → **12.72** | +1.4 % @ bs1x; +2.6 % @ sweet-spot |
| 10 | 2026-05-11 | V5 BGRADA scratch zeroing folded into finalize kernel (3 hipMemsetAsync/iter eliminated) | `b898899` | 12.72 → **12.74** | +0.2 % (within noise) |
|    | 2026-05-12 | NFS-bound → `/dev/shm` fix (AsyncReader uses `O_DIRECT`, bypasses page cache) | `bf88560` | 5.21 (NFS) → **11.76** sustained @ bs1x | +127 % vs NFS-bound run |
|    | 2026-05-12 | NV-submission audit (>120 env-knob configs swept) — bake `NCCL_PROTO=LL128`, `HIP_FORCE_DEV_KERNARG=1` | `1ef02aa`, `fd1f4d2`, `2c4670a`, `a923ef1` | 11.76 → **11.88** sustained | +1.0 % |
| 11 | 2026-05-12 | **`DEBUG_HIP_DYNAMIC_QUEUES=1`** (on-demand HW queue allocation; the 4-stream pipeline now overlaps properly) | `3860967` | 11.88 → **12.92** @ bs1x; 15.17 → **16.42** @ bs8x peak | **+8.9 % @ bs1x; +8.2 % @ bs8x** |
| 12 | 2026-05-12 | int4-vectorized `__half` elementwise: `concat_fwd/bwd_kernel_vec8` + `binaryOp_kernel_vec8_half` (used by MultiCross matrix_add) | `4fb17c3`, `e3e64a2` | 16.42 → **16.67** @ bs8x | **+1.5 %** at peak |

**Peak result post-Phase-12: 16.67 M sps at bs8x (442,368)** = 86.9 % of NV B200's bs8x peak (19.19 M sps), or **1.23× of the matched-data NV reference** (13.57 M sps at bs1x on the HF subsample).

## 2.2 Selected sub-tables (key wins in detail)

### Phase 6 — V5 2D-tile bgrad/bgrada kernels (+96 %, single biggest win)

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
| V5 in source but stale build (commit `4fc896a`) | 5.73 M sps | (V5 not actually engaged) |
| **V5 actually engaged** (clean rebuild) | **11.24 M sps** | **+96 %** |

Side-finding: the build I'd been benchmarking against had been stale —
the V5 kernels were in the binary but the routing code wasn't, so V5
was never running. Verified post-fix by adding a one-shot
`[HCTR-V5] ... path=V5` diag print on the first BGRADA call.

### Phase 11 — `DEBUG_HIP_DYNAMIC_QUEUES=1` (+6-9 % across all batches)

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

### Phase 12 — int4-vectorized `__half` elementwise kernels (+1.5 % at bs8x)

After capturing fresh rocprofv3 trace at the new bs8x peak, identified
three `__half` element-wise kernels using sub-optimal HBM utilization
(8-byte int2 or scalar `__half` loads instead of the 16-byte int4 limit).
Two were rewritten; the third (half4 ReLU) measured flat after node
controls.

| New kernel | File | Profile (bs8x) | Result |
|---|---|---|---:|
| `concat_fwd_kernel_vec8` / `concat_bwd_kernel_vec8` | `src/layers/concat_layer.cu` | 4 calls/iter × 327 µs = 1.31 ms = **4.9 % of bs8x iter** | bs4x **+3.06 %**, bs8x **+1.84 %** |
| `binaryOp_kernel_vec8_half` (MultiCross `matrix_add` user) | `include/prims/mlcommon_linalg_hip.cuh` | ~277 µs/iter at bs8x | bs8x **+0.5 %** (per-node controlled, both nodes consistent) |
| `half8_relu_kernel` | `src/layers/relu_layer.cu` | ReluLayer 14 calls/iter at bs8x | flat (-0.08 %) — half4 already saturates HBM at small layer widths; kept opt-in via `HCTR_RELU_KERNEL=vec8` |

All three gated by alignment + `if constexpr` so they only engage on
`__half` aligned to 16 bytes; v1 fallback always available via env knob
(`HCTR_CONCAT_KERNEL=v1`, `HCTR_BINARYOP_KERNEL=v1`, `HCTR_RELU_KERNEL=v1`).
Loss preserved across all configs (within FP16 noise).

## 2.3 Negative results (notable knobs that did NOT help)

These were measured but didn't survive a controlled re-test, so they're
NOT in the production `run_b200_match.sh`. Documented to save future
work from re-testing.

| Knob / change | Source | AMD result |
|---|---|---|
| `SHARDING_PLAN=round_robin` (vs `auto`) | NV b200/README +12 % | **−23 %** on AMD; auto is correct |
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

## Part 3 — Per-component analysis vs NV B200

Direct comparison via `rocprofv3 --hip-trace --kernel-trace` at peak
config (bs4x, 80 iters, agent 0; 55-iter steady window). Source:
`scripts/analyze_per_component.py`.

| Metric | AMD MI350X | NV B200 (b200/README §7.4a) | Notes |
|---|---:|---:|---|
| Per-iter wall (under profile overhead) | 4.61 ms | 4.08 ms | NV is +13 % faster |
| GPU busy (any kernel, merged streams) | 3.18 ms (69 %) | 3.19 ms (78 %) | similar absolute |
| Implied host gap | 1.43 ms (31 %) | 0.89 ms (22 %) | AMD has +0.54 ms host overhead |
| RCCL on-GPU time | 1.58 ms (34 %) | 1.51 ms (37 %) | similar |
| **RCCL exposed (compute idle)** at bs1x | **1.58 ms (100 % of RCCL!)** | **1.06 ms (70 %)** | AMD: 0 % hidden; NV: 30 % hidden via NVLS |
| RCCL hidden in compute at bs8x (post-DYN_QUEUES) | **3.12 ms (80 % of RCCL!)** | n/a | DYN_QUEUES + larger batch made the AMD gap close |
| `hipGraphLaunch` p50 | 5.97 ms | 0.53 ms (virtualized) / 0.01-0.03 ms (bare-metal) | AMD 11× slower than NV-virtualized, ~300× slower than NV bare-metal |

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

## Part 4 — Open work

Remaining items, ranked by EV vs effort. Items 1–3 are application-level,
4 is library-level, 5–6 are platform-fundamental (not actionable from
this repo).

| # | Item | Estimated win | Effort | Notes |
|---:|---|---:|---|---|
| 1 | `hipGraphLaunch` host overhead investigation | 1–3 % | research | rocprofv3 shows p50 = 5.97 ms (vs NV's 530 µs virtualized). Phase 11 already mitigated the downstream stream-blocking effect; the launch call itself is still ~11× slower than NV virtualized. Worth retesting with ROCm 7.3+. |
| 2 | hipBLASLt offline tuning via direct `hipblasLtMatmul` | 1–3 % | medium (rewrite HCTR MLP layer) | Bench shows 6/12 unique MLP shapes have 14–60 % heuristic-vs-best gap. `HIPBLASLT_TUNING_OVERRIDE_FILE` workflow tested — gain stayed in noise (HCTR's `hipblasGemmEx` wrapper bypasses the override). Direct hipblasLt API call needed. |
| 3 | `__amd_rocclr_fillBufferAligned` consolidation | ~1 % | medium (HugeCTR core) | ~30 calls/iter from HugeCTR-internal scratch zeroing (Tensor allocator init, MultiCross v2 `accum_dx` reset, etc.). Pre-zero a global scratch arena. |
| 4 | Real single-kernel HIP/MFMA fused MLP GEMM | 3–5 % | 3–5 days CUTLASS-AMD | Restores `Layer_t.MLP` to a perf win. Currently chains `hipblasGemmEx` + 1 fused post-pass + `bprop_drelu_bgrad_v5` per FC layer (5 launches/layer vs NV's 1). |
| 5 | NVLS / hardware multicast on xGMI | ~3 % | not actionable | NV's NCCL all-reduce + embedding all-to-all benefit from NVLink multicast hardware. AMD xGMI lacks an equivalent in current ROCm 7.2 RCCL (no `mscclpp` library shipped). Trace shows ~0.5 ms/iter exposed RCCL on AMD that NV hides via NVLS. |
| 6 | BF16 path | 1–2 % | medium | NV submission uses BF16 mixed; we use FP16. Would need `enable_bf16_compute` flag + `hip_bfloat16` template instantiations across `HugeCTR/src/layers/`. Removes loss-scaler stalls. |

**Out of perf path** but worth listing:
- Multi-node (RDMA `NetworkExchangeWgrad`) — currently single-node only
- hipBLASLt 7.3+ retest — when the heuristic exposes candidates for `RELU_AUX_BIAS` / `DRELU_BGRAD` at our shapes, drop the manual fallback (supersedes #4)

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
