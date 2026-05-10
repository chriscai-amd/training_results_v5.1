#!/usr/bin/env python3
"""
Convert preprocessed Criteo day_0 .npy files into HugeCTR's RawAsync MULTI-HOT
binary layout (the format NVIDIA's MLPerf submission consumes).

Per-sample layout (576 bytes for the default DLRM-DCNv2 MULTI_HOT_SIZES):
    <i32 label>                         #   4 B
    <f32 dense * 13>                    #  52 B
    <i32 keys[ sum(MULTI_HOT_SIZES) ]>  # 520 B   (130 keys at default config)
    -------------------------------------------
    total                               # 576 B / row

We synthesise the multi-hot expansion from our single-hot Criteo day_0:
for slot i with MULTI_HOT_SIZES[i] == k, the k keys for that slot are derived
deterministically from the single-hot id:
    keys[0] = id
    keys[j] = (id * P_j) % slot_size[i]            for j in 1..k-1
where P_j is a fixed odd 32-bit prime per offset j. This is NOT statistically
equivalent to NVIDIA's Meta-style hashing of the original Criteo categorical
fields (which we don't have on this node since we only converted day_0), but
it gives a deterministic, reproducible 576-B record stream that HugeCTR's
RawAsync reader accepts and that exercises the multi-hot embedding path
end-to-end.

Inputs:
  /apps/chcai/criteo_data/npy/day_0_{labels,dense,sparse}.npy
Outputs:
  /apps/chcai/criteo_data/hugectr_bin_mh/{train,val}_data.bin
"""
import argparse
import os
import sys
import time
import numpy as np

# NVIDIA MLPerf DLRM-DCNv2 default multi-hot expansion sizes (sum=130).
DEFAULT_MULTI_HOT_SIZES = [3, 2, 1, 2, 6, 1, 1, 1, 1, 7, 3, 8, 1, 6,
                           9, 5, 1, 1, 1, 12, 100, 27, 10, 3, 1, 1]

# MLPerf published per-slot embedding cardinalities.
DEFAULT_SLOT_SIZE = [
    40000000, 39060, 17295, 7424, 20265, 3, 7122, 1543, 63, 40000000,
    3067956, 405282, 10, 2209, 11938, 155, 4, 976, 14, 40000000,
    40000000, 40000000, 590152, 12973, 108, 36,
]

NUM_SLOTS = 26
NUM_DENSE = 13

# Fixed primes for synthetic multi-hot expansion (per-offset).
PRIMES = np.array([
    0x9E3779B1, 0x85EBCA6B, 0xC2B2AE35, 0x27D4EB2F, 0x165667B1, 0xD3A2646D,
    0x4F6CDD1D, 0xBA1F9C5B, 0xCE1F18B7, 0x6E5B0C03, 0x3243F6A9, 0xB5297A4D,
    0x0CF566AB, 0x68E31DA5, 0x6F4E0853, 0x7AAACE17, 0xA4093822, 0x299F31D1,
    0x082EFA98, 0xEC4E6C89, 0x452821E6, 0x38D01377, 0xBE5466CF, 0x34E90C6D,
    0xC0AC29B7, 0xC97C50DD, 0x3F84D5B5, 0xB5470917, 0x9216D5D9, 0x8979FB1D,
    0xD1310BA9, 0x98DFB5AD,
] * 4, dtype=np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy-dir",     default="/apps/chcai/criteo_data/npy")
    ap.add_argument("--out-dir",     default="/apps/chcai/criteo_data/hugectr_bin_mh")
    ap.add_argument("--day",         default="day_0")
    ap.add_argument("--train-frac",  type=float, default=0.95)
    ap.add_argument("--max-samples", type=int,   default=0)
    ap.add_argument("--slot-size",   type=int,   nargs="+", default=None)
    ap.add_argument("--multi-hot",   type=int,   nargs="+", default=None,
                    help="Per-slot multi-hot expansion sizes "
                         "(default = NVIDIA MLPerf DLRM-DCNv2)")
    args = ap.parse_args()

    slot_sizes = args.slot_size or DEFAULT_SLOT_SIZE
    mh_sizes   = args.multi_hot or DEFAULT_MULTI_HOT_SIZES
    if len(slot_sizes) != NUM_SLOTS:
        print(f"[FAIL] slot-size needs {NUM_SLOTS} entries, got {len(slot_sizes)}")
        sys.exit(1)
    if len(mh_sizes) != NUM_SLOTS:
        print(f"[FAIL] multi-hot needs {NUM_SLOTS} entries, got {len(mh_sizes)}")
        sys.exit(1)

    total_keys = sum(mh_sizes)
    bytes_per_row = 4 + NUM_DENSE * 4 + total_keys * 4

    os.makedirs(args.out_dir, exist_ok=True)
    train_path = os.path.join(args.out_dir, "train_data.bin")
    val_path   = os.path.join(args.out_dir, "val_data.bin")

    print(f"[..] mmaping {args.npy_dir}/{args.day}_*.npy")
    L = np.load(f"{args.npy_dir}/{args.day}_labels.npy", mmap_mode="r")
    D = np.load(f"{args.npy_dir}/{args.day}_dense.npy",  mmap_mode="r")
    S = np.load(f"{args.npy_dir}/{args.day}_sparse.npy", mmap_mode="r")

    N = L.shape[0]
    if args.max_samples and args.max_samples < N:
        print(f"[..] capping at {args.max_samples} rows (full file has {N})")
        N = args.max_samples
    n_train = int(N * args.train_frac)
    n_val   = N - n_train
    n_train -= n_train % 64
    n_val   -= n_val % 64

    print(f"[ok] N={N}  train={n_train}  val={n_val}")
    print(f"[ok] sum(MULTI_HOT_SIZES) = {total_keys}  bytes/row = {bytes_per_row}")
    print(f"[ok] slot sizes (head): {slot_sizes[:6]} ... total={sum(slot_sizes):,}")

    slot_mod = np.array(slot_sizes, dtype=np.uint32)
    mh_arr   = np.array(mh_sizes,   dtype=np.int32)

    # Pre-compute slot offsets in the flat keys array.
    slot_offsets = np.cumsum([0] + list(mh_sizes[:-1]))

    def emit(out_path, start, end):
        chunk = 1 << 14
        total_rows = end - start
        with open(out_path, "wb") as f:
            t0 = time.time()
            written = 0
            for s0 in range(start, end, chunk):
                s1 = min(s0 + chunk, end)
                k = s1 - s0

                row_buf = np.empty((k, 1 + NUM_DENSE + total_keys), dtype=np.int32)
                row_buf[:, 0] = L[s0:s1].reshape(-1).astype(np.int32, copy=False)
                row_buf[:, 1:1 + NUM_DENSE] = (
                    D[s0:s1].astype(np.float32, copy=False).view(np.int32))

                sp_u32 = S[s0:s1].astype(np.uint32, copy=False)  # (k,26)

                # Synthesise multi-hot keys per slot.
                for slot in range(NUM_SLOTS):
                    base_id  = sp_u32[:, slot]                   # (k,) uint32
                    mh       = mh_sizes[slot]
                    mod      = slot_mod[slot]
                    off      = 1 + NUM_DENSE + slot_offsets[slot]
                    # First key = the original id (mod slot size).
                    row_buf[:, off] = (base_id % mod).view(np.int32)
                    # Remaining keys = mixed via per-offset primes.
                    for j in range(1, mh):
                        mixed = (base_id * PRIMES[j]) % mod
                        row_buf[:, off + j] = mixed.view(np.int32)

                f.write(row_buf.tobytes())
                written += k
                if written % (1 << 17) == 0 or s1 == end:
                    rate = written / max(time.time() - t0, 1e-6)
                    print(f"  wrote {written:>10}/{total_rows} rows  {rate/1e3:.0f}k rows/s")

        sz = os.path.getsize(out_path)
        expected = total_rows * bytes_per_row
        ok = "OK" if sz == expected else f"MISMATCH (expected {expected})"
        print(f"[{ok}] {out_path} = {sz:,} bytes ({sz/1e9:.2f} GB), {total_rows:,} rows")

    print(f"\n[..] writing TRAIN -> {train_path}")
    emit(train_path, 0, n_train)
    print(f"\n[..] writing VAL   -> {val_path}")
    emit(val_path, n_train, n_train + n_val)
    print("\n[done]")


if __name__ == "__main__":
    main()
