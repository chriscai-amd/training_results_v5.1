"""Empirical item-frequency profiling of a Step-1 sparse npy file.

Reads day_N_sparse.npy (shape [N_rows, 26], int32) and for each of the 26
sparse-feature columns reports:

  - unique ID count
  - top-1, top-10, top-100 cumulative frequency (Pareto curve)
  - rough Zipf exponent alpha (least-squares fit on log-log of rank vs freq)

The fit is used by gen_zipfian_bin.py to synthesise data that matches the
access pattern of a real Criteo trace, so HugeCTR's auto/round_robin
planner and cuda graph capture see realistic frequencies.

Usage:
    python profile_sparse_freq.py --sparse data/processed_5b/day_0_sparse.npy
"""
from __future__ import annotations

import argparse
import json
import time
import numpy as np


TABLE_SIZE_ARRAY = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
    40000000, 40000000, 40000000, 590152, 12973, 108, 36,
]
NUM_TABLE = len(TABLE_SIZE_ARRAY)


def fit_zipf(counts_sorted: np.ndarray) -> tuple[float, float]:
    """Return (alpha, R2) of best-fit Zipf via log-log linear regression.

    counts_sorted: counts in decreasing order (so rank 1 has highest count).
    Zipf: count(rank) ∝ 1 / rank^alpha, so log(count) = -alpha*log(rank) + c.
    We fit on the high-count head where the law actually holds; for tables
    smaller than 100 IDs we fit on all observed.
    """
    n = len(counts_sorted)
    if n < 2:
        return float("nan"), float("nan")
    fit_range = min(max(50, n // 10), 10_000)
    fit_range = min(fit_range, n)
    rank = np.arange(1, fit_range + 1, dtype=np.float64)
    cnt = counts_sorted[:fit_range].astype(np.float64)
    log_r = np.log(rank)
    log_c = np.log(np.maximum(cnt, 1))
    # least-squares slope = -alpha
    A = np.column_stack([log_r, np.ones_like(log_r)])
    (slope, intercept), _, _, _ = np.linalg.lstsq(A, log_c, rcond=None)
    alpha = -slope
    pred = slope * log_r + intercept
    ss_res = np.sum((log_c - pred) ** 2)
    ss_tot = np.sum((log_c - np.mean(log_c)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(alpha), float(r2)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sparse", required=True,
                   help="Path to day_N_sparse.npy (shape [N, 26], int32)")
    p.add_argument("--out-json", default="",
                   help="Optional path to write per-table profile as JSON")
    p.add_argument("--head", type=int, default=10_000_000,
                   help="Sample at most this many rows for fitting (full file is fine)")
    args = p.parse_args()

    sparse = np.load(args.sparse, mmap_mode="r")
    print(f"loaded {args.sparse}: shape={sparse.shape} dtype={sparse.dtype}")
    if sparse.shape[1] != NUM_TABLE:
        raise SystemExit(f"expected {NUM_TABLE} columns, got {sparse.shape[1]}")
    n_rows = min(args.head, sparse.shape[0])
    print(f"using first {n_rows:,} rows\n")

    profile = []
    t0 = time.time()
    print(f"{'tbl':>3s}  {'cap':>10s}  {'unique':>10s}  {'%cap':>6s}  "
          f"{'p50_freq':>10s}  {'p99_freq':>10s}  {'top1%':>7s}  "
          f"{'top10%':>7s}  {'alpha':>6s}  {'R2':>6s}")
    for tbl in range(NUM_TABLE):
        col = np.ascontiguousarray(sparse[:n_rows, tbl])
        # use bincount-like via np.unique counts
        unique, counts = np.unique(col, return_counts=True)
        order = np.argsort(-counts)
        counts_sorted = counts[order]
        n_unique = len(unique)
        cap = TABLE_SIZE_ARRAY[tbl]
        cum = np.cumsum(counts_sorted)
        total = cum[-1]
        top1 = int(np.searchsorted(cum, total * 0.01) + 1)  # rank covering 1% of mass
        top10 = int(np.searchsorted(cum, total * 0.10) + 1)
        # top1%/top10% as % of total mass captured by top n IDs
        pct_top1 = float(cum[max(0, n_unique // 100 - 1)]) / total * 100 if n_unique > 0 else 0
        pct_top10 = float(cum[max(0, n_unique // 10 - 1)]) / total * 100 if n_unique > 0 else 0
        alpha, r2 = fit_zipf(counts_sorted)
        p50 = int(np.median(counts_sorted))
        p99 = int(np.percentile(counts_sorted, 99))
        print(f"{tbl:>3d}  {cap:>10,}  {n_unique:>10,}  "
              f"{n_unique/cap*100:>5.1f}%  {p50:>10}  {p99:>10}  "
              f"{pct_top1:>6.1f}%  {pct_top10:>6.1f}%  "
              f"{alpha:>6.2f}  {r2:>6.2f}")
        profile.append(dict(
            table=tbl, cap=cap, unique=n_unique,
            alpha=alpha, r2=r2,
            top1pct_mass=pct_top1, top10pct_mass=pct_top10,
            p50_freq=p50, p99_freq=p99,
        ))
    print(f"\nprofiled {NUM_TABLE} tables in {time.time() - t0:.1f}s")

    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"wrote profile JSON -> {args.out_json}")


if __name__ == "__main__":
    main()
