"""Critical-path analysis of an nsys trace: for each NCCL kernel, classify
whether it sits on the critical path (no concurrent compute on any other
stream of the same GPU) or whether it's parallelizable (compute is
running concurrently). Also dumps the stream layout and a 1-iter ASCII
timeline so we can SEE whether HugeCTR's captured CUDA graph schedules
NCCL on independent streams (overlap-able) or serialized (forced).

Usage:
    python critical_path_nsys.py --db b200_1x8_rr_full.sqlite [--gpu 0]
                                 [--iter-len-ms 4.05] [--n-iters 5]
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import Counter, defaultdict


def classify(name: str) -> str:
    n = name.lower()
    if "ncclkernel" in n or "ncclkern" in n or "rcclkern" in n or n.startswith("nccl") or "nccldevkernel" in n:
        return "comm"
    if "memcpy" in n or "memset" in n:
        return "copy"
    return "compute"


def merge_intervals(ivs):
    if not ivs:
        return []
    ivs = sorted(ivs)
    out = [list(ivs[0])]
    for s, e in ivs[1:]:
        if s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap_dur(s, e, merged):
    if not merged:
        return 0
    lo, hi = 0, len(merged)
    while lo < hi:
        m = (lo + hi) // 2
        if merged[m][1] <= s:
            lo = m + 1
        else:
            hi = m
    total = 0
    while lo < len(merged) and merged[lo][0] < e:
        total += max(0, min(e, merged[lo][1]) - max(s, merged[lo][0]))
        lo += 1
    return total


def find_train_window(rows, density_threshold=100, bin_ns=200_000_000):
    """Return (start_ns, end_ns) of the densely-NCCL part of the trace."""
    nccl = [(s,) for s, e, _, _, n in rows if classify(n) == "comm"]
    if not nccl:
        return None, None
    bins = Counter(s[0] // bin_ns for s in nccl)
    dense = sorted(b for b, c in bins.items() if c >= density_threshold)
    if not dense:
        return None, None
    return dense[0] * bin_ns, (dense[-1] + 1) * bin_ns


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--gpu", type=int, default=0,
                   help="Analyze this single GPU (default: 0)")
    p.add_argument("--iter-len-us", type=float, default=4050,
                   help="Approximate iter length in microseconds")
    p.add_argument("--n-iters", type=int, default=3,
                   help="How many iters of timeline to show")
    p.add_argument("--top-streams", type=int, default=12)
    p.add_argument("--width", type=int, default=120, help="Timeline width in chars")
    p.add_argument("--iter-skip", type=int, default=200,
                   help="Skip first N iters of training window before timeline")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.deviceId, k.streamId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        """
    ).fetchall()
    print(f"Loaded {len(rows):,} kernel events from {args.db}", file=sys.stderr)

    win_start, win_end = find_train_window(rows)
    if win_start is None:
        print("No training window found; using full trace.", file=sys.stderr)
        win_start = min(r[0] for r in rows if r[0] is not None)
        win_end = max(r[1] for r in rows if r[1] is not None)
    rows = [r for r in rows
            if r[0] is not None and r[1] is not None
            and r[1] >= win_start and r[0] <= win_end and r[2] == args.gpu]
    if not rows:
        print(f"No events on GPU {args.gpu} in training window.")
        return
    print(f"Training window {(win_end - win_start)/1e6:.0f} ms wide; "
          f"GPU {args.gpu} has {len(rows):,} kernels in window")

    # ---- 1) Per-stream busy time / kernel count / category mix
    per_stream = defaultdict(list)
    stream_cat = defaultdict(Counter)
    stream_dur_us = Counter()
    for s, e, _, sid, name in rows:
        cat = classify(name)
        per_stream[sid].append((s, e, cat, name))
        stream_cat[sid][cat] += 1
        stream_dur_us[sid] += (e - s) / 1000

    print()
    print(f"=== Streams on GPU {args.gpu} (top {args.top_streams} by busy time) ===")
    print(f"{'stream':<12} {'kern':>7} {'compute':>9} {'comm':>7} {'copy':>6} "
          f"{'busy_ms':>10} {'top_kernel':<60}")
    for sid, dur_us in stream_dur_us.most_common(args.top_streams):
        kc = stream_cat[sid]
        # most-frequent kernel name on this stream
        names = Counter(n for _, _, _, n in per_stream[sid])
        top_name, top_count = names.most_common(1)[0]
        print(f"{sid:<12} {sum(kc.values()):>7d} {kc.get('compute',0):>9d} "
              f"{kc.get('comm',0):>7d} {kc.get('copy',0):>6d} "
              f"{dur_us/1e3:>10.2f} {top_name[:60]}")

    # ---- 2) Critical-path analysis: for each NCCL kernel, is there any
    #        compute on ANY OTHER stream in its time window?
    comm_evts = [(s, e, sid, name) for s, e, _, sid, name in rows if classify(name) == "comm"]
    # Per-stream merged compute intervals (for quick overlap query)
    compute_per_stream = defaultdict(list)
    for s, e, _, sid, name in rows:
        if classify(name) == "compute":
            compute_per_stream[sid].append((s, e))
    compute_merged_per_stream = {sid: merge_intervals(ivs) for sid, ivs in compute_per_stream.items()}
    # Union of compute on streams != self
    def other_stream_compute_overlap(s, e, self_sid):
        # Build merged compute from all streams except self
        # (We approximate by summing per-stream overlap)
        total = 0
        for sid, merged in compute_merged_per_stream.items():
            if sid == self_sid:
                continue
            total += overlap_dur(s, e, merged)
        return total  # may double-count if multiple streams compute concurrently;
                     # we only need "is any > 0", so OK

    comm_total_us = 0
    comm_with_concurrent_compute_us = 0
    n_overlap_able_concurrent = 0
    n_critical_path = 0
    for s, e, sid, _ in comm_evts:
        dur_us = (e - s) / 1000
        comm_total_us += dur_us
        ovl_ns = other_stream_compute_overlap(s, e, sid)
        if ovl_ns > 0:
            comm_with_concurrent_compute_us += min(dur_us, ovl_ns / 1000)
            n_overlap_able_concurrent += 1
        else:
            n_critical_path += 1

    print()
    print(f"=== NCCL critical-path analysis (GPU {args.gpu}) ===")
    print(f"  total NCCL kernels             : {len(comm_evts):,}")
    print(f"  with concurrent compute on     : {n_overlap_able_concurrent:,} "
          f"({n_overlap_able_concurrent/max(1,len(comm_evts))*100:.1f}%)")
    print(f"  another stream                 ")
    print(f"  on critical path (none)        : {n_critical_path:,} "
          f"({n_critical_path/max(1,len(comm_evts))*100:.1f}%)")
    print(f"  total NCCL time                : {comm_total_us/1e3:.2f} ms")
    print(f"  NCCL time with concurrent comp : {comm_with_concurrent_compute_us/1e3:.2f} ms")
    print(f"  NCCL time on critical path     : {(comm_total_us - comm_with_concurrent_compute_us)/1e3:.2f} ms")

    # ---- 3) ASCII timeline of first N iters (after iter-skip)
    iter_len_ns = int(args.iter_len_us * 1000)
    timeline_start = win_start + args.iter_skip * iter_len_ns
    timeline_end = timeline_start + args.n_iters * iter_len_ns
    print()
    print(f"=== Timeline {args.n_iters} iters * {args.iter_len_us:.0f} us "
          f"(starting iter +{args.iter_skip}) ===")
    print(f"({timeline_start/1e6:.2f} -> {timeline_end/1e6:.2f} ms)")
    # Rasterize each stream into args.width-char row
    width = args.width
    iter_us = args.iter_len_us
    px_us = (iter_us * args.n_iters) / width
    char_for = {"compute": "#", "comm": "C", "copy": "."}
    print(f"  one char ≈ {px_us:.1f} us;  '#'=compute  'C'=NCCL  '.'=memcpy")
    print(f"  {'stream':<10}|{'-' * width}|")
    for sid, _ in stream_dur_us.most_common(args.top_streams):
        row = [' '] * width
        for s, e, _, name in per_stream[sid]:
            if e < timeline_start or s > timeline_end:
                continue
            cat = classify(name)
            ch = char_for[cat]
            i_s = max(0, int((s - timeline_start) / 1000 / px_us))
            i_e = min(width, int((e - timeline_start) / 1000 / px_us) + 1)
            for i in range(i_s, i_e):
                # Priority: comm > compute > copy (so we see comm when overlapping)
                cur = row[i]
                if cur == ' ':
                    row[i] = ch
                elif cat == 'comm' and cur != 'C':
                    row[i] = 'C'
                elif cat == 'compute' and cur == '.':
                    row[i] = '#'
        # Mark iter boundaries
        for k in range(args.n_iters + 1):
            i = int(k * iter_us / px_us)
            if 0 <= i < width and row[i] == ' ':
                row[i] = '|'
        print(f"  {str(sid)[:10]:<10}|{''.join(row)}|")


if __name__ == "__main__":
    main()
