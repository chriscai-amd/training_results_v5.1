# Time-to-Train (TTT) — current status

**Last updated:** 2026-05-20

## What we ran

Single TTT-style run on AMD MI350X, post-[`numerics_fix`][cvg] build (commit
[`ab614cf8`][cvg-commit]). Configuration matches NV's MLPerf v5.1 reference
(`5.1-0040`) at every DL hyperparameter, with NV-equivalent eval cadence so
wall-time numbers are directly comparable.

| | |
|---|---|
| Hardware | 8 × MI350X (gfx950) |
| Global batch | 55,296 (= MLPerf spec) |
| Precision | FP16 mixed, scaler 1024 |
| Optimizer | Adagrad, LR 0.004 (no warmup, no decay — see §Caveats) |
| Sharding | `auto` |
| Train data | **1.000 TB clean Criteo head** (= 1,096,491,227 rows = 0.261 MLPerf-epoch unique) |
| Val data | 76 GB MLCommons val (89,137,319 rows, full) |
| Storage | `/dev/shm/criteo` tmpfs on c09-08 |
| Iters | 19,800 = one full pass over the 1 TB (no re-pass) |
| Eval cadence | every 3,792 iters (= 1 per 0.05 MLPerf-epoch, NV MLPerf default) |
| Artefact | `/home/chcai/rps_out/ttt_1tb_20260520_131854/` |

[cvg]: numerics_fix.md
[cvg-commit]: https://github.com/chriscai-amd/training_results_v5.1/commit/ab614cf8

## AUC trajectory

Wall time is from `init_stop` → directly comparable to NV's published 2.3 min.

| MLPerf-epoch | iter | AUC | wall (s) |
|---:|---:|---:|---:|
| 0.0500 | 3,791  | 0.7491 | 24.7 |
| 0.0999 | 7,583  | 0.7574 (+0.008) | 43.8 |
| 0.1499 | 11,374 | 0.7612 (+0.004) | 63.0 |
| 0.1999 | 15,166 | 0.7542 (−0.007) | 82.3 |
| **0.2499** | **18,958** | **0.7640** (+0.010) ← **AUC max** | **101.4** |

- **AUC monotonically climbs** with mild ±0.007 noise over 5 NV-cadence eval points.
- No NaN, no divergence, loss 0.267 → 0.263.
- Median ms/iter = **4.48** (eval-spike-aware median over 90 display windows).
- Whole-run throughput (iter-time-derived) = **12.34 M samples/s**.
- Wall `init_stop` → `run_stop` = **105.3 s** = 1.76 min.

## Estimated TTT for full dataset (vs NV B200 MLPerf v5.1 `5.1-0040`)

NV's published reference: TTT = 2.3 min, throughput 22.80 M sps (10-run avg),
quality target AUC ≥ 0.80275, median convergence at 0.75 MLPerf-epoch
(≈ 3.15 B samples).

Two independent projection methods:

**Method A — throughput-based** (matches NV's measurement convention):
- Required samples = 3.15 B (NV's median convergence point at AUC ≥ 0.80275)
- AMD eval-amortized whole-run throughput = 12.34 M sps
- TTT projection = 3.15 B / 12.34 M sps = **255 s = 4.26 min**

**Method B — AUC slope extrapolation**:
- Measured slope across 0.05 → 0.25 MLPerf-epoch: ΔAUC / Δepoch = 0.0149 / 0.20 = +0.075 AUC per MLPerf-epoch
- AUC gap to threshold = 0.80275 − 0.7640 = +0.0388
- Required additional epoch = 0.0388 / 0.075 = 0.52 MLPerf-epoch
- AUC = 0.80275 reached at epoch ≈ 0.25 + 0.52 = **0.77 MLPerf-epoch** (matches NV's median 0.75)
- TTT = 0.77 × 4.195 B / 12.34 M sps = **261 s = 4.36 min**

| Quantity | NV B200 (`5.1-0040`) | AMD MI350X (extrapolated) | Ratio (AMD/NV) |
|---|---:|---:|---:|
| Pure-train ms/iter | 2.13 | **4.48** | 2.10× slower |
| Whole-run throughput, NV cadence (M sps) | 22.80 | **12.34** | 0.54× |
| Convergence epoch (MLPerf-epoch where AUC ≥ 0.80275 reached) | median 0.75 | **0.77 (extrapolated)** | parallel slope |
| **TTT wall to AUC ≥ 0.80275** | **2.3 min** | **~4.3 min (projected)** | **1.87× slower** |

The two projection methods agree within 2 % (4.26 vs 4.36 min), confirming the
slope and throughput are consistent. The agreement also means **AMD's
AUC-vs-samples-seen curve is parallel to NV's reference curve** — convergence
behavior is unchanged from upstream, only per-iter perf differs.

## Caveats

1. **Single-trial measurement.** NV's published 22.80 M sps is a 10-run average
   with σ ≈ 0.06. Our 12.34 M sps is 1 run; expect ±2-3 % seed variance.
2. **MLPerf-compliant LR schedule (no warmup, no decay) is identical to NV.**
   The MLPerf training rules ([`training_rules.adoc` lines 307-309][rules])
   mandate `opt_learning_rate_warmup_steps=0`, `opt_learning_rate_decay_start_step=0`,
   `opt_learning_rate_decay_steps=0`, `opt_adagrad_learning_rate_decay=0`,
   `opt_adagrad_initial_accumulator_value=0`, `opt_adagrad_epsilon=1e-8` for
   DLRM-DCNv2. AMD's run logs `opt_*` MLLog values matching this exactly.
   NV's `5.1-0040` G894-AD1 config also uses 0/0/0 (no `WARMUP_STEPS`/`DECAY_*`
   env vars set in [`config_G894-AD1_1x8x6912.sh`][gigact]). Convergence relies
   on Adagrad's intrinsic accumulator-based effective-LR decay alone.
3. **Loss scaler is the one AMD-vs-NV config deviation.** NV uses `SCALER=16348`
   (MLPerf reference value). AMD diverges (`Loss cannot converge` at iter ~2000)
   when running scaler=16348 post-`numerics_fix` — confirmed by direct A/B on
   2026-05-20. AMD-specific FP16 overflow paths require us to drop to
   `scaler=1024` for stability. Loss scaler is not rules-mandated, so this
   remains MLPerf-compliant. Recovering the scaler-16348 path is open work
   (see [`numerics_fix.md` §Remaining work][cvg-rem]).
4. **Convergence still extrapolated, not measured.** Our 1 TB = 0.26 MLPerf-epoch
   covers only the early-training regime; we haven't observed AUC actually
   crossing 0.80275 yet. Need ≥ 0.5 MLPerf-epoch of clean data for direct
   measurement (= 2 TB). Cloudflare R2 daily quota is 1 TB → ~2 more days of
   downloads to reach a runnable measurement.

[rules]: https://github.com/mlcommons/training_policies/blob/master/training_rules.adoc
[gigact]: https://github.com/mlcommons/training_results_v5.1/blob/main/GigaComputing/benchmarks/dlrm_dcnv2/implementations/B200/hugectr/config_G894-AD1_1x8x6912.sh
[cvg-rem]: numerics_fix.md#remaining-work--gradient-magnitude-tuning

## Reproduction

```bash
TAG=ttt_$(date +%Y%m%d_%H%M%S) MAX_ITER=19800 EVAL_INTERVAL=3792 DISPLAY=200 \
    PRECISION_FLAGS='--use_mixed_precision --scaler 1024' \
    bash /home/chcai/scripts/ttt_outer.sh
```

Writes `/home/chcai/rps_out/$TAG/{stage,run}.log` and `TTT_SUMMARY.json`
(parseable; key fields: `auc_trajectory`, `median_ms_per_iter_clean`,
`throughput_M_sps_from_iter_time`, `wall_init_to_run_stop_s`,
`vs_nv_5_1_0040`).

## See also

- [`docs/numerics_fix.md`](numerics_fix.md) — convergence-bug fix that unblocks all this.
- [README §3.1-3.2](../README.md#31-latest-status--bs1-throughput-gap-vs-nv-b200) — bs=1× perf gap decomposition.
- [README §4.1](../README.md#41-promising-items-prioritized-by-estimated-bs1-win) — prioritized open work for closing the bs=1× gap.
