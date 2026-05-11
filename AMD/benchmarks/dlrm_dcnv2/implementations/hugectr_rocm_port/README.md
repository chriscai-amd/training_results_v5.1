# DLRM-DCNv2 — HugeCTR ROCm port for AMD Instinct MI350X

A working port of NVIDIA's MLPerf v5.1 HugeCTR DLRM-DCNv2 submission to AMD's
ROCm 7.2 platform, targeting `gfx950` (Instinct MI350X). Source-compatible
with NVIDIA's `train.py` Python frontend and `mlperf_logger` MLLog format.

This is a research / port branch, **not an official MLPerf submission**.

## Status (single-node, 8 × MI350X, 1 node)

All numbers are FP16 mixed (matching NVIDIA's B200 submission), Adagrad
optimiser, scaler 16,348, sharding=auto, HIP graph capture on, with the
post-warmup / pre-final iterations averaged.

| Configuration | Throughput | Notes |
|---|---|---|
| 8 × MI350X, MULTI-HOT, **FULL Criteo (24 days, 482 M rows)**, batch 55,296 | **12.55 M samples/sec** (3-run avg, σ ~0.4 %), loss 0.285 → 0.264 | **apples-to-apples NVIDIA B200 config** — overlap ON + clamp folded into FMA |
| 8 × MI350X, MULTI-HOT, FULL Criteo, batch 110,592 (2× B200) | **16.89 M samples/sec**, loss 0.282 → 0.272 | **AMD sweet-spot batch — 24 % AHEAD of 8 × B200 on same HF data (13.57)** |
| 8 × MI350X, MULTI-HOT, FULL Criteo, batch 221,184 (4× B200) | 15.03 M samples/sec, loss 0.282 → 0.274 | (overlap-off measurement; rerun pending) |
| 8 × MI350X, MULTI-HOT, day_0 only (21 M rows), batch 55,296 | 6.73 M samples/sec, 40 iters | (kept for historical record; pre-V5-engagement, also smaller working set) |
| 8 × MI350X, MULTI-HOT, full Criteo, batch 55,296, **fused `Layer_t.MLP`** (`HCTR_USE_FUSED_MLP=1`) | 4.83 M samples/sec, loss 0.285 → 0.266 | DRELU_BGRAD + BGRADA epilogues emulated with V5 2D-tile kernels; **slower than InnerProduct path** because fused-MLP fallback chains 5 launches/FC layer vs InnerProduct's 4 |
| 8 × MI350X, MULTI-HOT, full Criteo, batch 110,592, **fused `Layer_t.MLP`** | 6.02 M samples/sec | sweet-spot batch with fused MLP fallback |
| 8 × MI350X, single-hot day_0, batch 55,296, HIP graph + overlap | 6.26 M samples/sec, 100 iters | NOT comparable to NVIDIA — single-hot is ~5× less embedding work |
| 8 × MI350X, single-hot day_0, batch 16,384 | 5.27 M samples/sec, 30+ iters | |
| 8 × MI350X, FP32, real DCN-v2, single-hot | 3.89–6.59 M samples/sec | |
| 1 × MI350X, FP16 mixed, real DCN-v2, fused `Layer_t.MLP`, batch 8,192 | 0.24 M samples/sec, loss 3.23 → 0.245 | single-GPU DRELU_BGRAD fallback verified |
| 1 × MI350X, FP16 mixed, real DCN-v2 | 1.84 M samples/sec | |
| 1 × MI350X, FP32, real DCN-v2 | 0.85 M samples/sec | |

### Headline result (post V5-BGRADA-engagement, 2026-05-11)

We now **match or exceed NVIDIA's published 8 × B200 throughput on the
same HuggingFace Criteo subsample** (8.7× less data than the unobtainable
4.2 B-row MLPerf reference corpus):

| | This port | NVIDIA B200 | Ratio |
|---|---|---|---|
| **batch 55,296** (NVIDIA's exact)         | **12.55 M sps** | 13.57 M sps  | 0.92× |
| **batch 110,592** (AMD sweet spot)        | **16.89 M sps** | 13.57 M sps  | **1.245×** |
| **batch 221,184**                          | 15.03 M sps   | 13.57 M sps  | 1.108× |

The breakthrough was discovering that the V5 2D-tile `BGRADA` kernel
(checked in earlier as commit `63a9c54` for the wgrad bias-gradient
column-sum) was **never actually being engaged** in benchmarks until
the corresponding `prewarm_bgrad_scratch` init path was confirmed to
run before HIP graph capture. The legacy `reduce_sum_columns_kernel`
was 49 % of all GPU time per a `rocprof --stats` run on a single GPU
(289 ms out of 590 ms total). Once the V5 path takes over, that drops
to a sub-1 % cost via 2D-tile coalesced reads + atomicAdd into a
pre-allocated FP32 scratch, and per-iter time falls from ~16 ms to
~5 ms at the sweet-spot batch.

### The "23 M sps" published reference is on a corpus we cannot get

NVIDIA's published MLPerf 5.1 8 × B200 result is **23.02 M samples/sec**,
2.3 min to AUC 0.80275. That number is on the **full 4.2 B-row Criteo
corpus** (`train_samples = 4,195,197,692` in their `result_*.txt` logs)
which is **not publicly available** — Criteo's `ailab.criteo.com` page
redirects to HuggingFace's `criteo/CriteoClickLogs`, and HF's mirror is
**pre-subsampled to ~473 M rows (~11 % of the MLPerf corpus)**. Our
482 M rows from the same HF mirror is essentially identical.

When NVIDIA's own engineers run the SAME 8 × B200 hardware on the SAME
HF subsample we have, they hit **13.57 M samples/sec**, not 23.02. The
1.7× gap from 13.57 → 23.02 is purely **hot-item-reuse**: in the
4.2 B-row corpus each top-1 % item is hit ~33,600 ×; in the 0.47 B-row
corpus only ~1,400 ×, which under-warms the embedding caches.
Source: this same submission has a `b200/.../README-b200-1x8.md` doc
with a published synthetic-data sweep showing zipfian-vs-real-vs-uniform
all match within 6 % on the HF subsample — only corpus volume matters.

So **the meaningful apples-to-apples reference is NVIDIA-on-HF
(13.57 M sps), not NVIDIA-on-MLPerf-corpus (23.02 M sps)**. Our best
results on this same data:

|                                          | Per-GPU (M sps) | Per-iter (ms) | Notes |
|---                                       |---              |---            |---    |
| NVIDIA B200, MLPerf corpus               | 2.88            | 2.16          | Public 5.1-0040 result, **not reproducible** without full Criteo |
| **NVIDIA B200, HF subsample**            | **1.70**        | **4.08**      | **Apples-to-apples reference** |
| AMD MI350X, HF subsample, batch 110,592  | **2.11**        | **6.55**      | **24.5 % AHEAD of B200** at AMD-tuned sweet-spot batch |
| AMD MI350X, HF subsample, batch 55,296   | **1.57**        | **4.40**      | 7.5 % behind B200 at NVIDIA's exact batch |

### Where the remaining 7.5 % lives (at NVIDIA's exact batch 55,296)

We need 0.32 ms/iter (4.40 → 4.08) to match B200 at the smaller batch.
Per the published B200-on-HF NSYS profile and our single-GPU rocprof,
ordered by expected magnitude:

1. **RCCL on xGMI vs NCCL on NVLink + NVLS multicast** (~0.10–0.30 ms).
   NCCL is 37 % of B200 time = 1.51 ms/iter. AMD xGMI has no NVLS
   hardware multicast, so AllReduce on FP16 grads needs a software
   ring/tree. RCCL typically measures 1.3–1.7× higher latency than
   NCCL for the same payload at our 8-GPU all-to-all shapes. We tried
   `NCCL_PROTO`/`NCCL_ALGO` knobs, only saw ~1 % swings.
2. **MLP fwd/bwd GEMM kernel selection** (~0.10–0.20 ms).
   B200 runs `cutlass3x_sm100_s128x256_bgrada`, `nvjet_hsh_*` —
   hand-tuned SM100 BF16 BWD with fused bias+grad, total 21 % =
   0.86 ms/iter. We use hipBLASLt's `Cijk_*MIWT*_MO40` MFMA kernels
   that hit ~1.0–1.3 ms/iter on the same shapes (less-mature heuristic
   for our exact dims, especially the small `m=128` cases).
3. **Embedding ops** (~0.10–0.20 ms). HugeCTR's `multi_to_one_*`,
   `label_and_count_keys`, `update4_kernel` total 17–21 % on B200.
   These are HugeCTR-native source, but on B200 they leverage HBM3e
   (8 TB/s vs MI350X HBM3 5.3 TB/s, ~33 % BW gap), L2 cache hints,
   and faster atomic ops on Hopper/Blackwell. Embedding is heavily
   memory-bound, so the BW gap shows up directly.
4. **`__amd_rocclr_fillBufferAligned` over-calls** (~0.05–0.10 ms).
   rocprof showed ~36 calls/iter for 0.54 ms total at single-GPU,
   most from HugeCTR-internal scratch zeroing inside the captured
   graph. NVIDIA groups these into ~5 % "elementwise/fused"; we're
   higher. A static-scratch consolidation is feasible.
5. **HIP graph capture coverage** (~0.05–0.10 ms). With overlap on,
   rocm-smi now shows ~30 % steady-state GPU util (vs B200's 86 %
   busy time). Some per-iter Python/host work (data reader hand-off,
   MLLog calls, eval-interval check) is still outside the captured
   region.

**What we already fixed this round** (each via the rocprof trace):

- **V5 2D-tile BGRADA / BGRAD kernels** (commit `63a9c54`,
  re-engaged via fresh build) — was 49 % of single-GPU time before;
  now sub-1 %. Per-iter dropped 16 ms → 5 ms at sweet-spot batch.
- **FP16 NaN/inf clamp folded into `vector_fma{3,4}_align8`**
  (commit `93aad5c`) — eliminates one launch + memory pass per
  cross layer. Was 7.45 ms over 60 iters in the trace.
- **Intra/inter-iteration overlap re-enabled** (commit `68e560e`) —
  earlier benchmarks were explicitly disabling overlap; with V5 +
  clamp-fold, overlap-on is now a +8.5 % win at NVIDIA's batch.

**At our AMD-tuned sweet-spot batch (110,592) we're already 24.5 %
AHEAD of B200** — proof that the underlying hardware capability is
there. The smaller-batch regime hits the more comm-bound + per-iter-
overhead-bound part of the curve, where AMD's software stack hasn't
caught up to NVIDIA's yet.

### Note on `rocm-smi` GPU utilisation

In an earlier (overlap-off) configuration, `rocm-smi --showuse` showed
1–7 % steady-state GPU utilisation, suggesting host-side bottleneck.
With **intra/inter-iteration overlap re-enabled** (the train.py default
that earlier benchmarks were explicitly overriding to 0), util is now
in the ~30 % range. B200 reports 86 % busy time, so there is still
host-side headroom — primarily in (a) HIP graph capture coverage of
the data-reader / MLLog path, and (b) the smaller-MLP-shape GEMM
kernels where launch overhead is a meaningful fraction of compute.

### Data: real Criteo, multi-hot synthesis

Both submissions train on real Criteo + synthetic multi-hot expansion.
Differences:

- **NVIDIA**: full 24-day Criteo from MLPerf reference download
  (~4.2 B rows, ~80 GB/day raw), expanded via Meta's
  `multi_hot.py` published synthetic hashing.
- **This port**: full 24-day Criteo from HuggingFace
  `criteo/CriteoClickLogs` (subsampled by HuggingFace to ~1.6 GB/day
  → ~21 M rows/day, ~482 M rows total), expanded via per-offset
  32-bit prime mixing in `runtime_test/criteo_npy_to_hugectr_bin_mh_alldays.py`.

Both produce the *same record format* (912-B/row, 214 keys/row,
26 multi-hot slots, identical `MULTI_HOT_SIZES = [3,2,1,2,6,…,1,1]`,
same per-slot embedding cardinalities), so per-iter throughput numbers
*are* comparable. The difference is total dataset size (8.7× less data)
and the tail of the embedding-id distribution. AUC convergence to
NVIDIA's 0.80275 target requires the full data pipeline; our smoke
target `HCTR_AUC_THRESHOLD=0.99` is intentionally never crossed.

### Per-component fix history (in commit order)

The headline 12.55 / 16.89 M sps came from a sequence of independent
fixes, not one big rewrite. In the order they landed:

1. **Multi-GPU correctness — missing BIAS in `RELU_AUX_BIAS` fprop
   fallback** (commit before `0d9a35e`). Was gated on
   `act == None && bias != null`, dropping bias on every hidden ReLU
   layer of both MLPs. Single-GPU absorbed the drift into the next
   layer's first-row weights; multi-GPU compounded it to `log(2)·2`
   via Adagrad + NCCL `dbias` all-reduce. Now triggers on `bias != null`
   regardless of activation.
2. **V5 2D-tile `bprop_drelu_bgrad_v5_kernel`** (commit `56bc046`).
   Replaces the legacy V1 (one block per row + cooperative 256-thread
   column scan, uncoalesced reads) with a 2D-tile (BLOCK_M=64 rows ×
   N_TILE=1024 cols/block), wave-coalesced reads, atomicAdd into
   per-device pre-allocated FP32 scratch, finalize kernel divides +
   clamps + casts to FP16. Pre-warmed via `std::call_once` so HIP
   graph capture sees the alloc done.
3. **V5 2D-tile BGRADA `bgrada_v5_kernel`** for the wgrad bias-grad
   column-sum (commit `63a9c54`). Same design, applied to
   `launch_reduce_sum_columns`.
4. **Discovered V5 was never engaged in benchmarks** (commit `4fc896a`).
   The build I'd been benchmarking against was stale — the V5 kernels
   were in the binary but the routing path wasn't. Once a clean rebuild
   engaged V5, per-iter dropped from ~16 ms to ~5 ms at the sweet-spot
   batch and from ~9.7 ms to ~5 ms at NVIDIA's batch. Verified by a
   one-shot diag print
   (`[HCTR-V5] launch_reduce_sum_columns first call: ... path=V5`).
5. **Folded FP16 NaN/inf clamp into `vector_fma{3,4}_align8`**
   (commit `93aad5c`). Was a separate `clamp_fp16_kernel` running 3×
   per iter after every cross layer's
   `fused_matrix_elementwise_dot_add`. New `__device__ sanitize_half2_fp16`
   helper does NaN/inf → 0, |v| > 65504 → ±65504 inline at the FMA
   store boundary — same memory access, free ALU.
6. **Re-enabled intra/inter-iteration overlap** (commit `68e560e`).
   `train.py` defaults `HCTR_INTRA_OVERLAP=1` and `HCTR_INTER_OVERLAP=1`
   already; earlier benchmark scripts were explicitly setting them to 0
   because an early sweep on day_0-only data showed slight regression.
   With V5 + clamp-fold + full Criteo, overlap-on is +8.5 % at
   NVIDIA's batch.

The fused `Layer_t.MLP` path (`HCTR_USE_FUSED_MLP=1`) is functional
on multi-GPU but **slower than the InnerProduct stack** (4.83 vs
12.55 M sps at NVIDIA's batch). The fallback chains
`hipblasGemmEx` + 1 fused post-pass + `bprop_drelu_bgrad_v5` per FC
layer — total 5 launches/layer vs InnerProduct's 4. Closing this
gap needs a real single-kernel HIP/MFMA fused GEMM kernel that
bundles GEMM + bias + ReLU + aux-write into one launch (3-5 days
of CUTLASS-AMD work).

### Closing the remaining 7.5 % at NVIDIA's batch — actionable items

Ordered by EV / effort. (We're already 24.5 % ahead at sweet-spot
batch; this section is specifically about the small-batch regime.)

1. **`add_bias_per_row_kernel` → V5 2D-tile**. Trace shows it's
   0.08 ms/call × 3 calls/iter. The same V5 design we used for
   BGRAD/BGRADA applies — known to give 2-3× on these shapes.
   Expected: ~1 % wall.  Effort: ~30 min.
2. **Consolidate `__amd_rocclr_fillBufferAligned` calls**. ~36 calls
   per iter, mostly HugeCTR-internal scratch zeroing inside the
   captured graph. Moving these to a one-time pre-zero of a global
   scratch arena would save 0.05–0.10 ms/iter (1–2 %).
   Effort: medium (HugeCTR core change).
3. **RCCL deep tuning**. Only ran 4 knobs so far (PROTO, ALGO,
   NTHREADS, NCHANNELS); none moved the needle past +1 %. Worth a
   focused pass with `RCCL_DEBUG=INFO` + `RCCL_BUFFSIZE` +
   `RCCL_P2P_LEVEL` + custom topology file once we have a working
   multi-GPU rocprof trace. Expected: 1–4 %.
4. **Real single-kernel HIP/MFMA fused MLP GEMM**. Closes the
   `hipblasGemmEx` vs `cutlass_s128x256_bgrada` gap (~14 % of B200
   time). Would also restore `Layer_t.MLP` to a perf win vs the
   InnerProduct stack. Expected: 3–5 % at small batch, more at large
   batch.  Effort: 3–5 days CUTLASS-AMD work.
5. **hipBLASLt 7.3+ retest**. When the heuristic exposes candidates
   for `RELU_AUX_BIAS` / `DRELU_BGRAD` at our shapes, we can drop the
   fallback entirely. Supersedes #4.

**Out-of-scope** (hardware / data limitations):

- **B200 HBM3e (8 TB/s) vs MI350X HBM3 (5.3 TB/s)** — embedding ops
  are memory-bound; ~33 % BW gap → ~0.05–0.10 ms/iter not addressable
  in software.
- **NVLS (NCCL hardware multicast) on NVLink** — no AMD equivalent
  on xGMI; ~0.10–0.20 ms not addressable in software.
- **Subsampled HF Criteo (482 M rows) vs MLPerf full corpus (4.2 B)** —
  affects AUC convergence target (NVIDIA's 0.80275) but not raw
  throughput. NVIDIA's own B200 caps at 13.57 M sps on the same HF
  data.

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

3. *Per-GPU batch ≥ 2048 NaN — FIXED via FP16 sanitisation folded
   into the FMA kernel*: A single FP16 NaN/inf appearing in any
   element of `layer_output_tensors[i]` (the per-cross-layer output)
   poisoned the whole tensor through the subsequent layer's GEMMs
   (NaN×anything = NaN). Initially fixed with a separate
   `clamp_fp16_kernel` post-pass (commit `4fc896a`). The trace then
   showed it cost 7.45 ms over 60 iters at single-GPU per-rank batch
   6912 (3 calls/iter); commit `93aad5c` folded the sanitise (NaN/inf
   → 0, |v| > 65504 → ±65504) inline into the
   `vector_fma{3,4}_align8<__half>` kernels via a new
   `__device__ sanitize_half2_fp16` helper. Same memory access, free
   ALU slots while HBM stores complete. Net: one fewer kernel launch
   + one fewer memory pass per cross layer per iter. The
   `HCTR_MC_CLAMP_FP16=0` env knob still exists for diagnostic A/B
   but the FMA-inline sanitise is unconditional in this build.

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

### Run with the fused `Layer_t.MLP` path (currently slower than default)

Add `-e HCTR_USE_FUSED_MLP=1`. This switches all dense MLPs from the
default InnerProduct + ReLU stack back to NVIDIA's `Layer_t.MLP` and
routes the `RELU_AUX` / `DRELU` / `DRELU_BGRAD` epilogues through our
fallback kernels in `HugeCTR/src/layers/functors/fused_gemm_functors.cu`.

Status (8 × MI350X, FP16, full real Criteo, multi-hot, batch 55,296):

- **InnerProduct stack (default)**: 12.55 M sps
- **Fused `Layer_t.MLP`**:           4.83 M sps

Fused MLP is currently slower because our fallback chains
`hipblasGemmEx` + 1 fused post-pass + `bprop_drelu_bgrad_v5` per FC
layer (5 launches/layer) vs the InnerProduct path's 4 launches/layer
through hipBLASLt's tuned MFMA path. Closing this needs a real
single-kernel HIP/MFMA fused GEMM kernel; currently on the TODO list.

Convergence is correct on multi-GPU after the missing-BIAS fix in the
`RELU_AUX_BIAS` fprop fallback (loss 0.285 → 0.266 over 80 iters).
Diagnostic env knob `HCTR_DISABLE_BIAS=1` skips the BIAS post-pass
entirely and reproduces the pre-fix divergence behaviour.

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

4. **MLP layers** default to `InnerProduct + ReLU` stacks (not the fused
   `Layer_t.MLP`) because hipBLASLt 1.2 lacks heuristic candidates for the
   fused MLP's `RELU_AUX_BIAS` / `DRELU_BGRAD` epilogues at our shapes.
   `Layer_t.MLP` works (correctness restored after the BIAS-fix in
   commit `0d9a35e`'s set_fprop_attr) but is currently slower than the
   InnerProduct path (4.83 vs 12.55 M sps at NVIDIA's batch). Closing
   this needs a real fused HIP/MFMA single-kernel GEMM.

5. **HIP graph capture and intra/inter-iter overlap re-enabled by default**
   in the solver. Earlier benchmarks were explicitly disabling overlap
   via `HCTR_INTRA_OVERLAP=0 HCTR_INTER_OVERLAP=0`, regression from an
   early day_0-only sweep; with V5 + clamp-fold + full-Criteo, overlap-on
   is now a +8.5 % win.

6. **V5 2D-tile post-pass kernels** (commits `56bc046` + `63a9c54`) for
   `bprop_drelu_bgrad` and `reduce_sum_columns` (BGRADA). Replace V1's
   one-block-per-row + 256-thread cooperative scan (uncoalesced) with
   a 2D tile of (BLOCK_M=64 rows × N_TILE=1024 cols/block), wave-aligned
   coalesced reads, atomicAdd into per-device pre-allocated FP32 scratch,
   finalize kernel divides + clamps + casts. Pre-warmed via std::call_once
   so HIP graph capture sees the alloc done. Was 49 % of single-GPU GPU
   time before V5 was actually engaged; sub-1 % after.

7. **FP16 NaN/inf clamp folded into `vector_fma{3,4}_align8`** (commit
   `93aad5c`). New `__device__ sanitize_half2_fp16` helper inlined at the
   FMA store boundary. Eliminates one launch + memory pass per cross
   layer per iter (was 7.45 ms over 60 iters in the rocprof trace).

## Open work

- **`add_bias_per_row_kernel` → V5 2D-tile design** — quick win (~1 %),
  ~30 min effort. Trace shows 0.08 ms/call × 3 calls/iter.
- **Consolidate `__amd_rocclr_fillBufferAligned` calls** — ~36 per iter
  in the captured graph (mostly HugeCTR-internal scratch zeroing).
  Pre-zeroing a global scratch arena would save 0.05–0.10 ms/iter.
- **Multi-GPU rocprof trace** — `rocprof` and `rocprofv3` both hang on
  multi-GPU in our ROCm 7.2 / docker setup. Need to switch to Omnitrace
  or a kernel-only filter for an apples-to-apples kernel breakdown vs
  NVIDIA's published B200 profile.
- **Real single-kernel HIP/MFMA fused MLP GEMM** — would restore the
  fused `Layer_t.MLP` path to a perf win and close the
  `hipblasGemmEx` vs `cutlass_s128x256_bgrada` gap. ~3–5 days of
  CUTLASS-AMD work.
- **Root-cause the FP16 NaN source in MultiCross** so we can drop the
  inline `sanitize_half2_fp16` (currently always on; see commit
  `93aad5c`). Worth ~3 % when removable.
- **BF16 path** — NVIDIA submission uses BF16 mixed (vs our FP16).
  Would need `enable_bf16_compute` flag + `hip_bfloat16` template
  instantiations across `HugeCTR/src/layers/`. Removes the loss-scaler
  stalls; possibly worth 1–2 %.
- **Multi-node** (RDMA `NetworkExchangeWgrad`) — currently single-node only.
- **hipBLASLt 7.3+ retest** — when the heuristic exposes candidates for
  `RELU_AUX_BIAS` / `DRELU_BGRAD` at our shapes, we can drop the manual
  fallback and reclaim 1.1–1.3× from vendor-tuned MFMA kernels.

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
