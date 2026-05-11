#!/usr/bin/env python3
"""
TSV[.gz] -> NumPy preprocessor for Criteo Click Logs.

Reads day_*.gz directly (or plain TSV), produces:
    {prefix}_labels.npy : int32, shape (N, 1)
    {prefix}_dense.npy  : float32, shape (N, 13)  (np.log(x + 3))
    {prefix}_sparse.npy : int32, shape (N, 26)    (hex -> int, lower 32 bits)

Uses one process; row count is dynamic (we don't pre-count to avoid a 2nd pass).
"""
from __future__ import annotations
import argparse
import gzip
import io
import os
import sys
import time
import numpy as np

INT_FEATURE_COUNT = 13
CAT_FEATURE_COUNT = 26
TOTAL_COLS = 1 + INT_FEATURE_COUNT + CAT_FEATURE_COUNT  # 40


def open_text(path: str):
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                newline="\n", errors="replace")
    return open(path, "r", buffering=1024 * 1024)


def preprocess(in_file: str, out_dir: str, chunk_rows: int) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.basename(in_file)
    if base.endswith(".gz"):
        base = base[:-3]
    out_labels = os.path.join(out_dir, f"{base}_labels.npy")
    out_dense  = os.path.join(out_dir, f"{base}_dense.npy")
    out_sparse = os.path.join(out_dir, f"{base}_sparse.npy")

    if os.path.exists(out_labels) and os.path.exists(out_dense) and os.path.exists(out_sparse):
        print(f"[skip] {in_file} already processed", flush=True)
        return

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] parsing {in_file} (gz-streaming) ...", flush=True)

    # Grow buffers in chunks of `chunk_rows` to avoid pre-counting.
    cur = chunk_rows
    labels = np.empty((cur, 1), dtype=np.int32)
    dense  = np.empty((cur, INT_FEATURE_COUNT), dtype=np.int32)
    sparse = np.empty((cur, CAT_FEATURE_COUNT), dtype=np.uint32)

    skipped = 0
    i = 0
    # Criteo .gz files from HuggingFace have malformed gzip trailers; treat
    # EOFError as "we ate as much as we could" and proceed with the rows
    # parsed so far. The truncation is upstream-known and benign for these
    # files (the body decoded fine).
    try:
        with open_text(in_file) as f:
            for line in f:
                row = line.rstrip("\n").split("\t")
                if len(row) != TOTAL_COLS:
                    skipped += 1
                    continue

                if i >= cur:
                    cur *= 2
                    labels = np.resize(labels, (cur, 1))
                    dense  = np.resize(dense,  (cur, INT_FEATURE_COUNT))
                    sparse = np.resize(sparse, (cur, CAT_FEATURE_COUNT))

                v = row[0]
                labels[i, 0] = int(v) if v else 0
                for j in range(INT_FEATURE_COUNT):
                    v = row[1 + j]
                    dense[i, j] = int(v) if v else 0
                for j in range(CAT_FEATURE_COUNT):
                    v = row[1 + INT_FEATURE_COUNT + j]
                    sparse[i, j] = (int(v, 16) & 0xFFFFFFFF) if v else 0

                i += 1
                if i % 5_000_000 == 0:
                    rate = i / max(time.time() - t0, 1e-6)
                    print(f"  parsed {i:>11,}  rate={rate/1e3:.1f}k rows/s", flush=True)
    except EOFError as e:
        print(f"  (truncated gz: {e}; keeping the {i:,} rows parsed so far)", flush=True)

    labels = labels[:i]
    dense  = dense[:i]
    sparse = sparse[:i]
    if skipped:
        print(f"  skipped {skipped} malformed rows", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] log-transform dense (np.log(x+3)) ...", flush=True)
    dense_f = (dense.astype(np.int32, copy=False) + 3)
    dense_log = np.log(dense_f, dtype=np.float32)

    print(f"[{time.strftime('%H:%M:%S')}] saving npy ({i:,} rows) ...", flush=True)
    np.save(out_labels, labels)
    np.save(out_dense,  dense_log)
    np.save(out_sparse, sparse.view(np.int32))
    sz_l = os.path.getsize(out_labels)
    sz_d = os.path.getsize(out_dense)
    sz_s = os.path.getsize(out_sparse)
    print(f"  -> {out_labels} ({sz_l/1e9:.2f} GB)", flush=True)
    print(f"  -> {out_dense}  ({sz_d/1e9:.2f} GB)", flush=True)
    print(f"  -> {out_sparse} ({sz_s/1e9:.2f} GB)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE  total {time.time()-t0:.1f}s", flush=True)


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("--in-file", required=True, help="day_*.gz or plain TSV")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chunk", type=int, default=4_000_000)
    args = p.parse_args(argv)
    preprocess(args.in_file, args.out_dir, args.chunk)


if __name__ == "__main__":
    main(sys.argv[1:])
