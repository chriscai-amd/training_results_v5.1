# MultiCross v2 backward gradient corruption — root cause & fix

**Date:** 2026-05-19  
**Phase tag:** `TTT.cvg`  
**Affected file:** `hugectr_hip/HugeCTR/src/layers/functors/fused_gemm_functors.cu`  
**Symptom:** training loss plateaus at ≈ 0.28; evaluation AUC stuck at chance (≈ 0.5) for thousands of iterations on real Criteo data, while NV B200 reaches AUC ≈ 0.75 in 200 iterations on the same model.

## TL;DR

AMD's `GemmFunctor::operator()` fallback path uses `hipblasGemmEx`, which is **in-place** (`D = α·A·B + β·D` — `mat_d` is both the input C and the output D). NV's `cublasLtMatmul` is **out-of-place** (`D = α·A·B + β·C` with separate `C`/`D` pointers). The old AMD code passed only `mat_d` to `hipblasGemmEx`, so callers passing `mat_c != mat_d` (i.e. true out-of-place GEMMs) had `mat_c` silently ignored and the prior junk in `mat_d` used as the β·C term. The MultiCross v2 `dY_prev` GEMM is the one out-of-place GEMM in the hot path, so the cross-network gradient chain was being computed against garbage — embeddings never learned, top-MLP could only learn the base-rate logit, AUC stayed at chance.

Fix is ~10 lines: when `mat_c != mat_d && beta != 0` copy `mat_c → mat_d` with `hipMemcpyAsync` before the GEMM. In-place callers (≈ all hot-path GEMMs: MC fprop, dV/dU accumulation, MLP wgrad, BGRADA fused path) hit the short-circuit and pay nothing.

## Background — why this took weeks to find

We initially mis-diagnosed the plateau as a MultiCross weight-initialisation issue. Debug prints inside HIP-graph capture showed zero weights, which is a known capture-time `hipMemcpy` artifact (host-side reads during capture return zeros / undefined values); chasing this red herring cost the most time. Once we read weights outside the capture region they were correctly Xavier-initialised. Forward MC math was also fine.

The real bug only manifests in MC **backward**, only on one GEMM, and only because that GEMM happens to be out-of-place. All other GEMMs in the model (forward MC, all MLPs, all wgrad accumulations) are in-place, so they look fine.

## Root cause

`GemmFunctor<T>::operator()` (`fused_gemm_functors.cu` ≈ line 957) takes both `const T* mat_c` and `T* mat_d` to match NV's `cublasLtMatmul` signature. On AMD we route through `hipblasGemmEx` for any "plain" GEMM that hipBLASLt cannot heuristically handle on gfx950 (which is most MC shapes):

```cpp
hipblasGemmEx(handle, op_a, op_b, m, n, k,
              &alpha,
              mat_a, ..., lda,
              mat_b, ..., ldb,
              &beta,
              mat_d, ..., ldc,   // <- mat_d serves as BOTH input C and output D
              compute_type, algo);
```

The hipblasGemmEx contract is `D = alpha*A*B + beta*D`. It does not take a separate `C` operand. The old code passed only `mat_d`, so `mat_c` was completely unused.

Almost every MC v2 GEMM (and every MLP GEMM) passes `mat_c == mat_d` — they want in-place accumulation into the same buffer, e.g. `dV += XU^T * S0`. They were unaffected.

The one out-of-place GEMM is the per-cross-layer `dY_prev` GEMM in `MultiCrossBackwardFunctorv2`:

```
grad_tensors[i] = S1 · U^T  +  grad_tensors[i+1]
   mat_d         mat_a   mat_b      mat_c                 (alpha = beta = 1)
```

`grad_tensors[i]` (mat_d) is the gradient at the bottom of cross layer `i`; `grad_tensors[i+1]` (mat_c) is the gradient at the top of the same layer (= what the *next layer up* wrote). They are different tensors — the loop chains them. With the bug, the GEMM computed `grad_tensors[i] = S1·U^T + JUNK` (junk being last iter's value of `grad_tensors[i]`), so the chain through the cross network was corrupted.

Critically, the junk is roughly bounded (it's the prior iter's dX) so training does not produce NaN — it just loses signal through MC, and MC's `dX` is the dominant gradient pathway to the embeddings.

## Isolation experiments

All on real Criteo (HF subsample), 8 × MI350X, FP16, scaler = 16,348, default `HCTR_LR=0.004` Adagrad, MLPerf-spec global batch 55,296. Each row = one rebuild + run.

| # | Experiment | AUC @ 200 | Interpretation |
|---:|:---|---:|:---|
| 1 | Baseline (the broken path) | **0.500** | plateau, no learning |
| 2 | Disable all FP16 NaN/inf clamps (`sanitize_half2_fp16`, kFp16Max clamps in `fused_gemm_functors.cu`) | NaN at iter 0 | clamps were masking real upstream NaN bursts — but the convergence bug is upstream of them |
| 3 | Disable BGRADA bias-grad write (`HCTR_DISABLE_BGRADA=1`) | 0.500 | bias-grad path is not the bug |
| 4 | Lower scaler to 1 (`--scaler 1`) | 0.500 | scaler is not the bug |
| 5 | Remove MC layer entirely (`HCTR_SKIP_MC=1`, concat1 → top-MLP direct) | **0.706** | MC is *actively destroying* signal, not just inert |
| 6 | Force MC weights = 0 (`HCTR_MC_FORCE_ZERO=1`); math says MC is a pure residual passthrough | 0.500 | MC backward is broken even when weights are zero — fwd math irrelevant |
| 7 | Force MC weights = 0 **+** skip MC bwd entirely (just `hipMemcpyAsync(dY → dX)`) | 0.708 | confirms MC bwd is the failure mode |
| 8 | Force MC weights = 0 **+** disable the `dgrads_[0] = in_tensors[0]` aliasing | 0.500 | aliasing is not the bug |
| 9 | Force MC weights = 0 **+** replace ONLY the `dY_prev` GEMM with `memcpy(mat_d ← mat_c)` | 0.707 | **pinpoints the `dY_prev` GEMM as the corruptor** |

Step 9 is the bisection result that named the bug: replacing a single GEMM with a memcpy (math-equivalent when weights are zero) restores convergence.

## The fix

```cpp
// fused_gemm_functors.cu, GemmFunctor<T>::operator(), inside the
// hipblasGemmEx fallback branch (immediately after hipblasSetStream).
//
// hipblasGemmEx is in-place: D = alpha*A*B + beta*D. It uses mat_d as
// BOTH the input C and the output D. When mat_c != mat_d (out-of-place
// GEMM, e.g. MC v2 dY_prev GEMM has mat_c=grad_tensors[i+1] and
// mat_d=grad_tensors[i]), we must copy mat_c -> mat_d BEFORE the GEMM so
// the beta*C term sees the right accumulator. Without this, the prior
// junk in mat_d gets used as the C input and the gradient chain is
// corrupted -- root cause of the training plateau (loss 0.28, AUC 0.5)
// on AMD vs NV.
if (mat_c != mat_d && beta != 0.0f) {
  size_t copy_bytes = sizeof(T) * static_cast<size_t>(cublas_desc.saved_ldc) *
                      static_cast<size_t>(cublas_desc.saved_n);
  HCTR_LIB_THROW(hipMemcpyAsync(mat_d, mat_c, copy_bytes,
                                hipMemcpyDeviceToDevice, stream));
}
```

The `mat_c == mat_d` short-circuit means in-place callers pay nothing. The only path that pays is true out-of-place GEMMs with `beta != 0` — in the current model that's just `dY_prev` (one GEMM per cross layer per iter, 3 copies at `num_cross_layers=3`).

The `beta != 0` guard skips the copy when the prior `mat_d` is going to be ignored anyway (β=0 means the formula is `D = α·A·B`, no C term).

## Effect on convergence

8 × MI350X, FP16, real Criteo HF subsample, MLPerf-spec bs=55,296 (6,912/GPU), default `HCTR_LR=0.004` Adagrad, 1,500 train iters with eval every 500.

| Config | AUC @ 200 | AUC @ 800 | Notes |
|:---|---:|---:|:---|
| Pre-fix (broken) | 0.500 | 0.500 | flat plateau, no learning |
| Post-fix, scaler 16,348 (current default) | NaN | NaN | gradients now real but FP16-overflow at hi scaler |
| Post-fix, scaler 8,192 | 0.665 | 0.725 | stable for ~1000 iters then NaN |
| Post-fix, scaler 4,096 | 0.650 | NaN | |
| Post-fix, scaler 2,048 | 0.500 | NaN | unlucky seed; intermittent |
| **Post-fix, scaler 1,024** | **0.744** | **0.748** | MC contributes ~4 AUC pts over no-MC baseline (0.706) |

The scaler sweep makes the "previously-eaten gradients are now real" symptom obvious: with the bug the gradients flowing back through MC were near-zero junk, so any scaler value worked; post-fix the gradients are real, and the production default scaler is too high for FP16 to hold them at intermediate accumulators.

## Performance impact

| GEMM | Calls / iter / GPU | Bytes / copy (bs=1×) | Extra time / iter |
|:---|---:|---:|---:|
| MC `dY_prev` (out-of-place, β=1) | 3 (one per cross layer) | ~45 MB | ≈ 0.5 ms |
| All other GEMMs (in-place) | many | (short-circuited) | 0 |

≈ +0.5 ms / iter, ≈ +8 % on a 6 ms/iter steady-state — acceptable for the correctness gain. Easy follow-ups to recover most/all of it:

1. **Fuse the `mat_c → mat_d` copy into the next launched kernel** (the FMA at the start of the next bwd loop iter loads from `grad_tensors[i]` anyway — extend it to load from `mat_c` if a flag says so).
2. **Use `hipblasLtMatmul` for these specific shapes** if a working algo can be heuristically picked (`gfx950` hipBLASLt 1.2 returns zero candidates for most MC shapes today; revisit on hipBLASLt updates).
3. **Skip the copy when `mat_a` or `mat_b` is provably zero** (e.g. on iteration 0 when MC weights are tiny) — would need a runtime check.

## Remaining work — gradient-magnitude tuning

This fix is correctness-only; it does not change any hyperparameter. Because the previously-eaten backward gradients are now real, several knobs that were inertly correct under the bug now need to be re-tuned:

* **Scaler.** Production default `scaler=16,348` now overflows FP16 at intermediate accumulators. Drop to `scaler=8,192` or `scaler=1,024` with linear warmup.
* **LR warmup + poly decay.** NV's reference uses ~2,750 step linear warmup and degree-2 polynomial decay. AMD's default has both off (`warmup_steps=0`, `decay_steps=0`). Enable to match NV's reference trajectory.
* **Re-validate perf phases 15–18.** The broken MC bwd was ~0.5 ms / iter cheaper than the corrected path; the M sps numbers in Part 2's master timeline table reflect the buggy path and need a fresh A/B at the post-fix gradient magnitude.

This fix UNBLOCKS time-to-train (TTT). All previous TTT attempts were training on noise; we couldn't see any TTT lift from optimizer/LR tuning because the model wasn't learning.

## See also

* `hugectr_hip/HugeCTR/src/layers/functors/fused_gemm_functors.cu` — `GemmFunctor<T>::operator()` (line ≈ 1005, fallback branch)
* `hugectr_hip/HugeCTR/src/layers/multi_cross_layer.cu` — `MultiCrossBackwardFunctorv2<T>::operator()`, the `dY_prev` GEMM call
* `docs/cudagraph_analysis.md` — related ROCm-specific HIP-graph capture pitfalls (the source of the original misdiagnosis)
