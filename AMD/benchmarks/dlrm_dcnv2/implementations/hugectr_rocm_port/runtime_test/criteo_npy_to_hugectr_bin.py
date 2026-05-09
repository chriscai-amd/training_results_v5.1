#!/usr/bin/env python3
"""
Convert preprocessed Criteo day_0 .npy files into HugeCTR's RawAsync multi-hot
binary layout. Per sample (single-hot, 26 slots):
    <i32 label> <f32 dense * 13> <i32 key * 26>

Inputs:  /apps/chcai/criteo_data/npy/day_0_{labels,dense,sparse}.npy
Outputs: /apps/chcai/criteo_data/hugectr_bin/{train,val}_data.bin

Apply hash-modulo to bound the key space (MLPerf-published slot sizes total
~166M, but we can use smaller for a smoke test).
"""
import argparse
import os
import sys
import time
import numpy as np

# DLRM-DCNv2 / MLPerf default slot sizes (per-feature embedding cardinalities).
DEFAULT_SLOT_SIZE = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]
NUM_SLOTS = 26
NUM_DENSE = 13


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir",    default="/apps/chcai/criteo_data/npy")
    ap.add_argument("--out-dir",    default="/apps/chcai/criteo_data/hugectr_bin")
    ap.add_argument("--day",        default="day_0")
    ap.add_argument("--train-frac", type=float, default=0.95)
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Cap rows (0 = all rows). Useful for fast smoke tests.")
    ap.add_argument("--slot-size", type=int, nargs="+", default=None,
                    help="Override per-slot embedding sizes. Default = MLPerf published.")
    args = ap.parse_args()

    slot_sizes = args.slot_size or DEFAULT_SLOT_SIZE
    if len(slot_sizes) != NUM_SLOTS:
        print(f"[FAIL] slot-size must have {NUM_SLOTS} entries, got {len(slot_sizes)}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_data.bin")
    val_path   = os.path.join(args.out_dir, "val_data.bin")

    print(f"[..] mmaping {args.npy_dir}/{args.day}_*.npy")
    L = np.load(f"{args.npy_dir}/{args.day}_labels.npy", mmap_mode="r")  # (N,1) i32
    D = np.load(f"{args.npy_dir}/{args.day}_dense.npy",  mmap_mode="r")  # (N,13) f32 (log-transformed)
    S = np.load(f"{args.npy_dir}/{args.day}_sparse.npy", mmap_mode="r")  # (N,26) i32 (hash bytes)

    N = L.shape[0]
    if args.max_samples and args.max_samples < N:
        print(f"[..] capping at {args.max_samples} rows (full file has {N})")
        N = args.max_samples
    n_train = int(N * args.train_frac)
    n_val   = N - n_train
    # Round both down to nearest 64 for clean batching.
    n_train -= n_train % 64
    n_val   -= n_val % 64
    print(f"[ok] N={N}  train={n_train}  val={n_val}")
    print(f"[ok] slot sizes: {slot_sizes[:6]} ... (total={sum(slot_sizes):,})")

    slot_mod = np.array(slot_sizes, dtype=np.uint32)

    def emit(out_path, start, end):
        # Per row layout: label(i32) + dense(f32*13) + sparse(i32*26)
        chunk = 1 << 16
        bytes_per_row = 4 + NUM_DENSE * 4 + NUM_SLOTS * 4
        total_rows = end - start
        with open(out_path, "wb") as f:
            t0 = time.time()
            written = 0
            for s0 in range(start, end, chunk):
                s1 = min(s0 + chunk, end)
                # Build the row buffer
                lab = L[s0:s1].reshape(-1).astype(np.int32, copy=False)  # (k,)
                den = D[s0:s1].astype(np.float32, copy=False)            # (k,13)
                # Sparse keys: HugeCTR wants distinct embeddings per slot, not collide
                # across slots. Apply per-slot modulo to bound the key range.
                # The sparse npy was written via uint32 view of int32; reinterpret as uint32.
                sp_u32 = S[s0:s1].astype(np.uint32, copy=False)
                sp = (sp_u32 % slot_mod).astype(np.int32, copy=False)    # (k,26)

                # Pack one row at a time (slow path; OK for a smoke test).
                rows = np.empty((s1 - s0, 1 + NUM_DENSE + NUM_SLOTS), dtype=np.int32)
                rows[:, 0] = lab
                # Reinterpret f32 dense as int32 bits in-place
                rows[:, 1:1 + NUM_DENSE] = den.view(np.int32)
                rows[:, 1 + NUM_DENSE:] = sp
                f.write(rows.tobytes())
                written += (s1 - s0)
                if written % (1 << 18) == 0 or s1 == end:
                    rate = written / max(time.time() - t0, 1e-6)
                    print(f"  wrote {written:>10}/{total_rows} rows  {rate/1e3:.0f}k rows/s")
        sz = os.path.getsize(out_path)
        expected = total_rows * bytes_per_row
        ok = "OK" if sz == expected else f"MISMATCH (expected {expected})"
        print(f"[{ok}] {out_path} = {sz:,} bytes ({sz/1e9:.2f} GB), {total_rows:,} rows")

    print(f"\n[..] writing TRAIN -> {train_path}")
    emit(train_path, 0, n_train)
    print(f"\n[..] writing VAL -> {val_path}")
    emit(val_path, n_train, n_train + n_val)
    print("\n[done]")


if __name__ == "__main__":
    main()
