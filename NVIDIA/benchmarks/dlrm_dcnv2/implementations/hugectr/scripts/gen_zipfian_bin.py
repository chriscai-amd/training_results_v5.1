"""Generate train/val .bin files with per-table Zipfian sparse-index draws.

Compared to gen_synthetic_bin.py (uniform draws), this generates indices
that match the long-tail access pattern of real Criteo:

  - Per-table alpha values come from profile_sparse_freq.py output (or
    fallback constants if no profile JSON is supplied).
  - Indices are drawn from np.random.zipf(alpha) and clipped to [0, cap),
    which gives the most-popular IDs at index 0,1,2,... and a long tail.

The point is to make HugeCTR's auto/round_robin planner cost model and
cuda graph capture see realistic hot-item reuse over the full 40M-cap
embedding tables, even when we don't have the 4.2B-row corpus.

Usage:
    python gen_zipfian_bin.py --rows 2_200_000 \\
        --out /data/criteo_synth_zipf/train_data.bin --seed 42 \\
        --profile scripts/criteo_freq_profile.json
"""
from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np


TABLE_SIZE_ARRAY = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63,
    40000000, 3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14,
    40000000, 40000000, 40000000, 590152, 12973, 108, 36,
]
MULTI_HOT_SIZES = [
    3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6, 9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1,
]
NUM_DENSE = 13

# Fallback alpha (used if --profile not supplied). Based on profile_sparse_freq
# observation that big tables are α≈1.05 and small tables vary 0.9–4.0.
FALLBACK_ALPHA = [
    1.04, 1.22, 0.89, 1.47, 1.30, 8.48, 0.90, 1.22, 4.91,
    1.05, 1.19, 1.17, 2.89, 1.16, 1.25, 3.29, 1.25, 1.89, 3.03,
    1.04, 1.10, 1.05, 1.14, 0.92, 3.76, 3.94,
]


def sample_zipf(rng: np.random.Generator, alpha: float, cap: int, size: int) -> np.ndarray:
    """Draw `size` samples from Zipf(alpha) clipped to [0, cap).

    For α ≤ 1 numpy raises; we floor to 1.01 in that case (very long tail).
    For caps < 30 (e.g. table 5 with cap=3), Zipf is degenerate; use
    simple modulo-fold over [0, cap) which still preserves long-tail.
    """
    if alpha <= 1.0:
        alpha = 1.01
    if cap < 30:
        # zipf with α very high gives mostly 1s; just modulo-fold
        raw = rng.zipf(max(alpha, 1.5), size=size)
        return ((raw - 1) % cap).astype(np.int32)
    raw = rng.zipf(alpha, size=size)
    # zipf can return very large values; clip
    return np.minimum(raw - 1, cap - 1).astype(np.int32)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-rows", type=int, default=200_000)
    p.add_argument("--profile", default="",
                   help="JSON profile from profile_sparse_freq.py (per-table alpha). "
                        "If empty, uses FALLBACK_ALPHA constants.")
    args = p.parse_args()

    if args.profile:
        with open(args.profile) as f:
            prof = json.load(f)
        alphas = [float(x["alpha"]) for x in prof]
        if len(alphas) != len(TABLE_SIZE_ARRAY):
            raise SystemExit(f"profile has {len(alphas)} tables, expected {len(TABLE_SIZE_ARRAY)}")
    else:
        alphas = list(FALLBACK_ALPHA)

    print(f"per-table alpha (used for Zipf sampling):")
    for tbl, (cap, mh, a) in enumerate(zip(TABLE_SIZE_ARRAY, MULTI_HOT_SIZES, alphas)):
        print(f"  table {tbl:>2d}: cap={cap:>10,d}  multi-hot={mh:>3d}  alpha={a:.2f}")
    print()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total_sparse = sum(MULTI_HOT_SIZES)
    bytes_per_row = 4 + 4 * NUM_DENSE + 4 * total_sparse  # 912
    print(f"rows={args.rows:,}  cols=label+{NUM_DENSE}f32+{total_sparse}i32  "
          f"row={bytes_per_row}B  total={args.rows * bytes_per_row / 1e9:.2f} GB\n")

    t0 = time.time()
    written = 0
    with open(args.out, "wb") as fh:
        for chunk_start in range(0, args.rows, args.chunk_rows):
            n = min(args.chunk_rows, args.rows - chunk_start)
            dense = rng.standard_normal((n, NUM_DENSE)).astype(np.float32)
            sparse_cols = []
            for cap, mh, alpha in zip(TABLE_SIZE_ARRAY, MULTI_HOT_SIZES, alphas):
                s = sample_zipf(rng, alpha, cap, n * mh).reshape(n, mh)
                sparse_cols.append(s)
            sparse = np.concatenate(sparse_cols, axis=1)
            assert sparse.shape == (n, total_sparse)
            # Synthesise a learnable label signal so HugeCTR's BF16 loss path
            # doesn't blow up with NaN/Inf. Use a hash-mod-2 of the first index
            # of each big table XOR'd together; gives a deterministic 50/50 signal
            # that the model can actually learn (lookup → xor → linear is fine
            # for an MLP). Without this, the optimizer chases noise and FP16/BF16
            # overflows after a few hundred iters.
            big_table_first_cols = [
                1,                     # table 0 first col idx in sparse[]
                1 + 3 + 2 + 1 + 2 + 6 + 1 + 1 + 1 + 1 - 1,  # table 9
                # don't bother with all 5; 2 is enough to give a signal
            ]
            label = np.zeros(n, dtype=np.int32)
            for col in big_table_first_cols:
                label ^= (sparse[:, col] & 1).astype(np.int32)
            buf = np.empty((n, bytes_per_row // 4), dtype=np.int32)
            buf[:, 0] = label
            buf[:, 1:1 + NUM_DENSE] = dense.view(np.int32)
            buf[:, 1 + NUM_DENSE:] = sparse
            fh.write(buf.tobytes())
            written += n
            if chunk_start % (args.chunk_rows * 5) == 0:
                rate = written / max(1e-3, time.time() - t0)
                print(f"  written {written:,} / {args.rows:,}  ({rate / 1e6:.2f} M rows/s)")

    print(f"done in {time.time() - t0:.1f}s -> {args.out} ({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
