#!/usr/bin/env python3
"""
Convert preprocessed Criteo *.npy files (all 24 days OR a subset) into
HugeCTR's RawAsync MULTI-HOT binary layout, matching NVIDIA's MLPerf submission.

Per-sample layout (576 bytes for the default DLRM-DCNv2 MULTI_HOT_SIZES=130):
    <i32 label>                         #   4 B
    <f32 dense * 13>                    #  52 B
    <i32 keys[ sum(MULTI_HOT_SIZES) ]>  # 520 B   (130 keys at default config)
    -------------------------------------------
    total                               # 576 B / row

Splits: NVIDIA's MLPerf reference uses the LAST half of day_23 as the eval set
(~89 M samples). We follow the same convention here so the val_data.bin shape
matches the NVIDIA recipe.

Synthesis: for slot i with MULTI_HOT_SIZES[i] == k > 1, the k keys for that
slot come from the real Criteo single-hot id mixed via per-offset 32-bit
primes (deterministic, same scheme as NVIDIA's published synthetic multi-hot
expansion in spirit, just different prime constants).

Inputs:  /apps/chcai/criteo_data/npy/day_{0..23}_{labels,dense,sparse}.npy
Outputs: /apps/chcai/criteo_data/hugectr_bin_mh_full/{train,val}_data.bin
"""
import argparse
import os
import sys
import time
import numpy as np

DEFAULT_MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6,
                           9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]

DEFAULT_SLOT_SIZE = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]

NUM_SLOTS = 26
NUM_DENSE = 13

PRIMES = np.array([
    0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646D,
    0x4F6CDD1D, 0xBA1F9C5B, 0xCE1F18B7, 0x6E5B0C03, 0x3243F6A9, 0xB5297A4D,
    0x0CF566AB, 0x68E31DA5, 0x6F4E0853, 0x7AAACE17, 0xA4093822, 0x299F31D1,
    0x082EFA98, 0xEC4E6C89, 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6D,
    0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917, 0x9216D5D9, 0x8979FB1D,
    0xD1310BA9, 0x98DFB5AD,
] * 4, dtype=np.uint32)


def load_day(npy_dir, d):
    """Returns (labels, dense, sparse) memory-mapped per day, or None if missing."""
    paths = (
        f"{npy_dir}/day_{d}_labels.npy",
        f"{npy_dir}/day_{d}_dense.npy",
        f"{npy_dir}/day_{d}_sparse.npy",
    )
    if not all(os.path.exists(p) for p in paths):
        return None
    L = np.load(paths[0], mmap_mode="r")
    D = np.load(paths[1], mmap_mode="r")
    S = np.load(paths[2], mmap_mode="r")
    n = min(L.shape[0], D.shape[0], S.shape[0])
    return L[:n], D[:n], S[:n]


def emit_rows(f, L, D, S, slot_sizes, mh_sizes, slot_offsets, total_keys,
              start, end, chunk=1 << 14, t0=None, prefix=""):
    """Stream rows [start, end) from (L, D, S) into open file f."""
    if t0 is None:
        t0 = time.time()
    slot_mod = np.array(slot_sizes, dtype=np.uint32)
    written = 0
    for s0 in range(start, end, chunk):
        s1 = min(s0 + chunk, end)
        k = s1 - s0
        row_buf = np.empty((k, 1 + NUM_DENSE + total_keys), dtype=np.int32)
        row_buf[:, 0] = L[s0:s1].reshape(-1).astype(np.int32, copy=False)
        row_buf[:, 1:1 + NUM_DENSE] = (
            D[s0:s1].astype(np.float32, copy=False).view(np.int32))
        sp_u32 = S[s0:s1].astype(np.uint32, copy=False)

        for slot in range(NUM_SLOTS):
            base_id = sp_u32[:, slot]
            mh = mh_sizes[slot]
            mod = slot_mod[slot]
            off = 1 + NUM_DENSE + slot_offsets[slot]
            row_buf[:, off] = (base_id % mod).view(np.int32)
            for j in range(1, mh):
                mixed = (base_id * PRIMES[j]) % mod
                row_buf[:, off + j] = mixed.view(np.int32)
        f.write(row_buf.tobytes())
        written += k
        if written % (1 << 19) == 0 or s1 == end:
            elapsed = time.time() - t0
            print(f"  {prefix}wrote {written:>11,}/{end-start:,} rows  "
                  f"{written/elapsed/1e3:.0f}k rows/s", flush=True)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir",   default="/apps/chcai/criteo_data/npy")
    ap.add_argument("--out-dir",   default="/apps/chcai/criteo_data/hugectr_bin_mh_full")
    ap.add_argument("--days",      type=int, nargs="+",
                    default=list(range(24)),
                    help="Day indices to include in TRAIN. Day 23's last half goes to VAL.")
    ap.add_argument("--val-frac-of-day23", type=float, default=0.5,
                    help="Fraction of day_23 used for VAL (last N rows). Default = MLPerf 0.5")
    ap.add_argument("--slot-size", type=int, nargs="+", default=None)
    ap.add_argument("--multi-hot", type=int, nargs="+", default=None)
    args = ap.parse_args()

    slot_sizes = args.slot_size or DEFAULT_SLOT_SIZE
    mh_sizes   = args.multi_hot or DEFAULT_MULTI_HOT_SIZES
    if len(slot_sizes) != NUM_SLOTS or len(mh_sizes) != NUM_SLOTS:
        print(f"[FAIL] need {NUM_SLOTS} slot/mh entries"); sys.exit(1)

    total_keys = sum(mh_sizes)
    bytes_per_row = 4 + NUM_DENSE * 4 + total_keys * 4
    slot_offsets = np.cumsum([0] + list(mh_sizes[:-1]))

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_data.bin")
    val_path   = os.path.join(args.out_dir, "val_data.bin")

    # Discover available days
    available, missing = [], []
    for d in args.days:
        rec = load_day(args.npy_dir, d)
        (available if rec is not None else missing).append(d)
    print(f"[ok] days available: {available}")
    if missing:
        print(f"[!!] days missing : {missing}")

    print(f"[ok] sum(MULTI_HOT_SIZES) = {total_keys}  bytes/row = {bytes_per_row}")

    t0 = time.time()
    train_rows = 0
    val_rows = 0

    with open(train_path, "wb") as ftrain, open(val_path, "wb") as fval:
        for d in available:
            L, D, S = load_day(args.npy_dir, d)
            n = L.shape[0]
            print(f"[day_{d}] N={n:,}", flush=True)

            if d == 23:
                # Split day_23 between train and val (NVIDIA convention).
                n_val_d23 = int(n * args.val_frac_of_day23)
                n_val_d23 -= n_val_d23 % 64
                n_train_d23 = n - n_val_d23
                n_train_d23 -= n_train_d23 % 64
                print(f"[day_23] train rows {n_train_d23:,}  val rows {n_val_d23:,}")
                train_rows += emit_rows(
                    ftrain, L, D, S, slot_sizes, mh_sizes, slot_offsets,
                    total_keys, 0, n_train_d23, t0=t0, prefix=f"d23-train ")
                val_rows += emit_rows(
                    fval, L, D, S, slot_sizes, mh_sizes, slot_offsets,
                    total_keys, n - n_val_d23, n, t0=t0, prefix=f"d23-val   ")
            else:
                # All rows go to train, rounded to 64.
                n -= n % 64
                train_rows += emit_rows(
                    ftrain, L, D, S, slot_sizes, mh_sizes, slot_offsets,
                    total_keys, 0, n, t0=t0, prefix=f"d{d:>2}-train ")

    sz_train = os.path.getsize(train_path)
    sz_val   = os.path.getsize(val_path)
    expected_train = train_rows * bytes_per_row
    expected_val   = val_rows   * bytes_per_row
    print(f"\n[ok] train_data.bin = {sz_train:,} B ({sz_train/1e9:.2f} GB), {train_rows:,} rows  "
          f"{'OK' if sz_train == expected_train else 'MISMATCH'}")
    print(f"[ok] val_data.bin   = {sz_val:,} B ({sz_val/1e9:.2f} GB), {val_rows:,} rows  "
          f"{'OK' if sz_val == expected_val else 'MISMATCH'}")
    print(f"\n[done] total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
