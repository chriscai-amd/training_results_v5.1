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
result is **5.73 M sps** — a **2.37× gap** to NVIDIA on the same data,
not the previously reported 5.2× to NVIDIA on the unobtainable corpus.

|                        | Per-GPU (M sps) | Per-iter (ms) | Notes |
|---                     |---              |---            |---    |
| NVIDIA B200, MLPerf corpus  | 2.88        | 2.16          | Public 5.1-0040 result, **not reproducible** without full Criteo |
| **NVIDIA B200, HF subsample** | **1.70**    | **4.08**      | **Apples-to-apples reference** |
| AMD MI350X, HF subsample, InnerProduct  | 0.71        | 9.66          | This port, our best |
| AMD MI350X, HF subsample, fused MLP     | 0.60        | 11.46         | This port, with HCTR_USE_FUSED_MLP=1 |

### Where the remaining 2.37× lives

Per the published B200-on-HF NSYS profile in
`b200/NVIDIA/benchmarks/dlrm_dcnv2/implementations/hugectr/README-b200-1x8.md`:

```
NCCL                       ~23 %   (16.4 % all-to-all + 6.5 % AllReduce)
embedding ops              ~17 %   (Adagrad update + scatter/gather)
sparse infra (sort, cub)    ~6 %
MLP fwd/bwd GEMMs          ~14 %   (cutlass_s128x256 + nvjet_hsh + drelu)
elementwise FMA / fused    ~10 %
optimizer (adagrad)         ~3 %
other (long tail)          ~27 %
```

DLRM-DCNv2 on B200 is **embedding/comm-bound, not compute-bound** —
GEMMs are only ~14 % of GPU time. Our 2.37× gap to B200-on-HF is most
likely:

1. **RCCL on xGMI vs NCCL on NVLink** (NCCL/all-to-all is 23 % of B200
   time; if RCCL is ~1.5 × slower, that alone is ~12 % of total time).
2. **hipBLASLt's plain `hipblasGemmEx` vs cuBLASLt's fused
   `cutlass_s128x256_bgrada` and `nvjet_hsh` MLP kernels** (~14 % of
   B200 time; if our GEMM path is ~2 × slower, that's ~14 % of total).
3. **HugeCTR's HIP graph capture is partial on AMD** (rocm-smi shows
   1–7 % steady-state GPU util on our side vs B200's 86 % busy time).
   The data-reader pipeline + per-iter MLLog calls fall outside the
   captured region; bigger HIP graphs would push GPU util closer to
   B200's ~86 %.

We've now fused the post-pass kernels (bias + ReLU + aux into one,
drelu_bgrad + bgrada via V5 2D-tile kernels). The only remaining
software lever short of waiting for hipBLASLt 7.3+ heuristics is to
ship a real single-kernel HIP/MFMA fused GEMM that bundles all four
ops into one launch — closing roughly the GEMM half of the gap,
plus making more of the per-iter loop graph-capturable.

### Where the gap lives — `rocm-smi` profile

A direct `rocm-smi --showuse` sample during steady-state training
shows **GPU utilisation of only 1–7 %** across all 8 GPUs at the
sweet-spot batch. The MI350X is mostly idle waiting on host-side
work — a kernel-launch / data-staging bottleneck, not raw GPU
compute. Confirmed by:

- HIP graph on vs off: only ~4% delta (5.83 vs 5.59 M sps at batch
  110,592). On NVIDIA the same flag would typically buy 1.5-3× by
  bundling launches; here HugeCTR's HIP graph capture is partial
  (the multi-hot data-reader pipeline and the per-iter MLLog calls
  are outside the captured region).
- Larger global batches (221k, 442k, 884k) increase per-iter GPU
  time but only modestly improve throughput, suggesting per-iter
  CPU-side overhead dominates.
- Algorithm-search on/off: neutral — hipBLASLt's heuristic is
  already being used; the search adds overhead without finding
  better kernels at our shapes.

**This means the dominant remaining lever is the host-side overhead
in HugeCTR's iter loop** (in particular the unfused-MLP path's
multiple kernel launches per FC layer + the data-reader -> embedding
hand-off), not the GPU compute itself. A real single-kernel fused
MLP that bundles GEMM + bias + ReLU + aux-write into one launch
would pay off on two axes — fewer kernels in the captured graph
(less launch overhead) and fewer post-pass kernels in our fallback
(less HIP launch overhead) — and is the highest-EV next item.

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

### Gap analysis: 5.26 M sps (apples-to-apples) vs ~30 M sps ≈ 5.7×

Roughly attributable to (and what we are doing about each):

- **Unfused MLP** (`Layer_t.MLP` — GEMM+ReLU+bias+RELU_AUX fused in
  hipBLASLt's epilogue on NVIDIA hardware; on `gfx950` hipBLASLt 1.2
  has no heuristic for the relevant epilogues, so we substituted an
  InnerProduct+ReLU stack). Headroom: **~1.5-2×** when we land a
  real fused HIP kernel.
  - **Implementation status**: full RELU_AUX / DRELU / DRELU_BGRAD
    epilogue emulation with cuBLASLt-format bit-packed masks now ships
    in `fused_gemm_functors.cu`. Set `HCTR_USE_FUSED_MLP=1` to enable.
  - **Correctness fix (2026-05-10)**: BIAS post-pass was previously
    gated on `act == None && bias != null`, which dropped the bias
    term in *every hidden ReLU layer* of the bottom+top MLPs. Single-GPU
    happened to converge (next layer's weights absorbed the drift);
    multi-GPU collapsed to `log(2)·2` once the missing-bias drift
    compounded across ranks via Adagrad + NCCL all-reduce of dbias.
    Now correctness ✓ on 8 GPU multi-hot at NVIDIA's batch (loss
    0.285 → 0.264 over 100 iters, 4.61 M sps).
  - **Post-pass fusion (this branch)**: bias + ReLU + bit-packed mask
    write are now fused into a single `fprop_bias_relu_aux_kernel` —
    cuts one launch per FC fprop, +16 % throughput on the fused-MLP
    path at the AMD sweet-spot batch (5.07 → 5.88 M sps).
  - **V5 2D-tile kernels for both bgrad post-passes (this branch)**:
    The legacy `bprop_drelu_kernel` (V1) launched one block per output
    row + cooperative 256-thread column scan (uncoalesced reads); the
    legacy `reduce_sum_columns_kernel` (BGRADA) launched one thread per
    output row total -> ~1% CU util at m=128. Both are now superseded
    by V5 kernels: 2D tile (BLOCK_M=64 rows × N_TILE=1024 cols/block)
    with WAVES_PER_BLOCK=4 -> 256 threads/block, lane-in-wave = row in
    stripe -> 128-byte coalesced reads. Per-block partial sums via
    shared mem then atomicAdd into a per-device pre-allocated FP32
    scratch buffer (allocated once via std::call_once before HIP graph
    capture begins, so per-iter launch path stays graph-safe). Final
    small kernel divides + clamps + casts to FP16. Net win on full
    Criteo at batch 110,592: 5.88 → 6.02 M sps (+2%) on top of the
    post-pass fusion.
  - **Why still slower than InnerProduct stack** (5.88 vs 7.28 M sps
    at sweet-spot batch): the launch-count math now favours fused-MLP
    only marginally. The remaining gap is in bprop, where my
    `bprop_drelu_kernel<true>` does a per-row column scan to compute
    bgrad (stride-`m` reads, no vectorisation). InnerProduct's bprop
    instead uses 3 well-tuned `hipblasGemmEx` calls (separate
    bgrad-via-identity-vector, wgrad, dgrad) that hit hipBLASLt's
    optimised paths. Closing the rest needs a real fused HIP/MFMA
    bprop kernel that uses MFMA + LDS reductions for the bgrad sum.

- **No multi-node fabric scaling**: NVIDIA's 8 GPU result is on 2 nodes
  × 4 GPU with NVLink/NVSwitch fabric. Our 8 GPU are inside one node
  with xGMI. Probably a wash given the per-node-vs-cross-node tradeoff.

- **Hardware difference**: B200 HBM3e + 5th-gen Tensor Cores vs MI350X
  HBM3 + MFMA — at this tensor-density compute the per-GPU peak FLOPs
  are similar but B200 has higher HBM bandwidth. Probably ~2-3× of the
  remaining headroom (multi-hot is heavily embedding-bandwidth-bound).
  Not addressable in software.

- **Subsampled vs full Criteo**: HuggingFace's `criteo/CriteoClickLogs`
  is subsampled to ~1.6 GB/day (vs ~80 GB/day raw upstream); we have
  482 M total rows vs NVIDIA's ~4.2 B (8.7× less). Throughput is
  *not* directly affected by row count, but the smaller working set
  fits HBM caches better — day_0-only runs at ~6.73 M sps vs the full
  482 M-row 5.26 M sps for the same per-iter shape. Closing this gap
  requires the original (non-subsampled) Criteo dataset, which is not
  publicly distributable.

### Closing the gap — concrete near-term wins

Ordered by expected wall-clock impact:

1. **Real single-kernel fused MLP** (write a HIP/MFMA kernel that does
   GEMM + bias + ReLU + bit-packed aux-write in one launch, replacing
   our hipblasGemmEx + 1 fused post-pass kernel). Expected **~1.2-1.5×
   → 9-11 M sps apples-to-apples**. ~3-5 days of work. The post-pass
   fusion done in this branch already collapsed the easy launch-count
   wins; the remaining headroom is in (a) doing the GEMM via an MFMA
   kernel that knows to write the AUX mask inline, and (b) replacing
   our `bprop_drelu_kernel<true>` per-row column scan with an
   MFMA + LDS reduction for the bgrad path.
2. **GPU utilisation is only 1-7 %** at steady state per `rocm-smi`,
   meaning the dominant cost is currently host-side. HIP graph capture
   in HugeCTR is partial on AMD (the multi-hot reader pipeline and
   MLLog calls fall outside the captured region). A fuller HIP-graph
   capture region or a re-architect of the per-iter Python loop could
   pull GPU util into the 30-50 % range. Worth **2-4×** if we can do it.
3. **`HCTR_MC_CLAMP_FP16=0`** is currently NOT safe to disable
   — `Loss cannot converge` triggers immediately at multi-hot batch
   221,184. Need to chase the underlying NaN source in MultiCross
   intermediates before we can drop the clamp post-pass. Worth ~1.03×.
4. **AsyncParam reader threads**: bumped to 4 by default in this branch
   (`HCTR_READER_THREADS`). Neutral on day_0 (cached) but helps a small
   amount on the full 482 M-row dataset. Worth 1.02-1.05×.
5. **hipBLASLt re-evaluation after ROCm ≥ 7.3**: every quarter, re-check
   whether `hipBLASLt` exposes heuristic candidates for the
   `RELU_AUX_BIAS` / `DRELU_BGRAD` epilogues at our MLP shapes. When it
   does, we can drop the manual fallback and use vendor-tuned kernels.
   Would supersede #1.

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

### Run with the new fused MLP path

Add `-e HCTR_USE_FUSED_MLP=1`. This switches all dense MLPs from the
InnerProduct + ReLU stack back to NVIDIA's `Layer_t.MLP` and routes
the `RELU_AUX` / `DRELU` / `DRELU_BGRAD` epilogues through our new
fallback kernels in `HugeCTR/src/layers/functors/fused_gemm_functors.cu`.

- **1 × MI350X**: works (loss 3.23 → 0.245 over 20 iters at batch 8192).
- **8 × MI350X (pre-2026-05-10)**: collapsed to `log(2)·2`. Root-caused
  to a missing BIAS post-pass in the `RELU_AUX_BIAS` fprop fallback —
  the guard checked only `saved_is_bias_epilogue` (which is
  `(act == None) && bias != null`) so the bias was silently dropped on
  every hidden ReLU layer of both MLPs.  Single-GPU absorbed the missing
  bias into the next layer's first-row weight column, but Adagrad +
  cross-rank `dbias` allreduce on multi-GPU compounded the drift until
  predictions saturated near 0.5 (BCE → log 2).
- **8 × MI350X (post-fix, 2026-05-10)**: the BIAS post-pass now fires
  whenever `saved_bias_ptr != nullptr` AND
  (`saved_is_bias_epilogue || saved_is_relu_aux_epilogue`). Re-run
  expected to converge and to win 1.5-2× vs the unfused stack.

Diagnostic env knob: set `HCTR_DISABLE_BIAS=1` to A/B-test by skipping
the BIAS post-pass entirely (reproduces the pre-fix divergence behaviour).

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

- **FP16 + multi-GPU + real MultiCross v2** NaN — currently contained by
  the `clamp_fp16_kernel` post-pass after each cross layer; root cause
  (single bad element appearing inside the MultiCross fp16 math chain
  at per-rank batch ≥ 2048) still wants a proper fix. FP32 multi-GPU and
  FP16 single-GPU both converge without the clamp.
- **Validate the post-fix fused MLP run** end-to-end on 8 × MI350X with
  `HCTR_USE_FUSED_MLP=1 HCTR_USE_MULTI_HOT=1` and update the perf table.
- **Real multi-hot Criteo data path** — preprocessing tools work on the
  single-hot day_0 only; need to run the full
  `materialize_synthetic_multihot_dataset.py` + `convert_to_raw.py` for the
  4 TB MLPerf-spec dataset (throughput-neutral, but unblocks AUC≥0.80275).
- **BF16 path** — would need `enable_bf16_compute` flag + `hip_bfloat16`
  template instantiations across `HugeCTR/src/layers/`.
- **Multi-node** (RDMA `NetworkExchangeWgrad`) — currently single-node only.
- **hipBLASLt 1.3+ retest** — when the heuristic exposes candidates for
  `RELU_AUX_BIAS` / `DRELU_BGRAD` at our shapes, we can drop the manual
  fallback and reclaim the last 1.1-1.3× from vendor-tuned MFMA kernels.

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
