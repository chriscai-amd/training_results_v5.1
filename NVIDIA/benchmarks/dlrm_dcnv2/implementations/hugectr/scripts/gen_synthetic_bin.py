"""Generate a synthetic train_data.bin / val_data.bin in HugeCTR raw format.

Each row is packed as:
  [int32 label]
  [13 × float32 dense]
  [Σ multi_hot_sizes int32 sparse]
= 4 + 52 + 520 = 576 B / row

Sparse indices are drawn uniformly in [0, TABLE_SIZE_ARRAY[i]) to span the
full vocab — this lets us isolate whether the perf gap vs MLPerf-reference
is data-related (smaller subsample → biased access patterns and
auto-planner cost-model miscalibration) or hardware/system-related.

Usage:
  python gen_synthetic_bin.py --rows 2000000 \\
      --out /data/criteo_synth_uniform/train_data.bin --seed 42
"""
from __future__ import annotations

import argparse
import os
import struct
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


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-rows", type=int, default=200_000)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    rng = np.random.default_rng(args.seed)
    total_sparse = sum(MULTI_HOT_SIZES)
    bytes_per_row = 4 + 4 * NUM_DENSE + 4 * total_sparse  # = 576
    print(f"rows={args.rows:,}  cols=label+{NUM_DENSE}f32+{total_sparse}i32  "
          f"row={bytes_per_row}B  total={args.rows * bytes_per_row / 1e9:.2f} GB")

    t0 = time.time()
    written = 0
    with open(args.out, "wb") as fh:
        for chunk_start in range(0, args.rows, args.chunk_rows):
            n = min(args.chunk_rows, args.rows - chunk_start)
            label = rng.integers(0, 2, size=n, dtype=np.int32, endpoint=False)
            dense = rng.standard_normal((n, NUM_DENSE)).astype(np.float32)
            sparse_cols = []
            for cap, mh in zip(TABLE_SIZE_ARRAY, MULTI_HOT_SIZES):
                sparse_cols.append(
                    rng.integers(0, cap, size=(n, mh), dtype=np.int32, endpoint=False)
                )
            sparse = np.concatenate(sparse_cols, axis=1)
            assert sparse.shape == (n, total_sparse)
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
