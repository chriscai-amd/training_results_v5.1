"""Compute compute-vs-comm breakdown (exposed vs hidden NCCL) from a nsys
SQLite database. Mirrors the analysis used for Kineto traces in
Primus-DLRM/scripts/analyze_comms.py.

Per-GPU algorithm:
    - Pull all GPU kernel events (CUPTI_ACTIVITY_KIND_KERNEL)
    - Classify: NCCL/RCCL → comm; others → compute
    - For each GPU, merge compute intervals, then for each comm interval
      compute the overlap → that's "hidden". (comm_dur - hidden) is "exposed".

Usage:
    python breakdown_nsys.py --db b200_1x8_iter50_v4.sqlite

The expected SQLite is produced by `nsys stats` (it auto-creates a .sqlite
next to the .nsys-rep on first run).
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import defaultdict


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


def overlap_us(a_s, a_e, merged):
    if not merged:
        return 0
    lo, hi = 0, len(merged)
    while lo < hi:
        m = (lo + hi) // 2
        if merged[m][1] <= a_s:
            lo = m + 1
        else:
            hi = m
    total = 0
    while lo < len(merged) and merged[lo][0] < a_e:
        total += max(0, min(a_e, merged[lo][1]) - max(a_s, merged[lo][0]))
        lo += 1
    return total


def classify(name: str) -> str:
    n = name.lower()
    if "ncclkernel" in n or "ncclkern" in n or "rcclkern" in n or n.startswith("nccl") or "nccldevkernel" in n:
        return "comm"
    return "compute"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT k.start, k.end, k.deviceId, s.value
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.demangledName
        """
    ).fetchall()
    print(f"Loaded {len(rows):,} kernel events from {args.db}")

    # Auto-detect training window via NCCL kernel density. Init does a brief
    # NCCL handshake (~few events) but training has hundreds of NCCL kernels
    # per iter; we keep only the densely-NCCL bins.
    nccl_times = [(s, e) for s, e, _, n in rows if classify(n) == "comm"]
    if nccl_times:
        from collections import Counter
        BIN_NS = 200_000_000  # 200 ms
        DENSE_THRESHOLD = 100  # events/bin
        bins = Counter(s // BIN_NS for s, _ in nccl_times)
        dense_bins = sorted(b for b, c in bins.items() if c >= DENSE_THRESHOLD)
        if dense_bins:
            win_start = dense_bins[0] * BIN_NS
            win_end = (dense_bins[-1] + 1) * BIN_NS
            print(f"Training window (NCCL-dense bins {dense_bins[0]}..{dense_bins[-1]}): "
                  f"{(win_end - win_start) / 1e6:.2f} ms wide")
            rows = [r for r in rows
                    if r[1] is not None and r[1] >= win_start and r[0] <= win_end]
            print(f"After filtering: {len(rows):,} events")

    per_gpu = defaultdict(lambda: {"compute": [], "comm": []})
    cat_total_us = defaultdict(int)
    name_total_us = defaultdict(int)

    for s, e, dev, name in rows:
        if e is None or s is None or e <= s:
            continue
        cat = classify(name)
        per_gpu[dev][cat].append((s, e))
        cat_total_us[cat] += (e - s) / 1000
        name_total_us[(cat, name)] += (e - s) / 1000

    summary = {"compute_us": 0, "comm_total_us": 0, "comm_exposed_us": 0, "comm_hidden_us": 0,
               "wall_us": 0}
    for dev, evts in per_gpu.items():
        compute_merged = merge_intervals(evts["compute"])
        comm_merged = merge_intervals(evts["comm"])
        union_merged = merge_intervals(evts["compute"] + evts["comm"])
        compute_us = sum(e - s for s, e in compute_merged) / 1000
        comm_us = sum(e - s for s, e in comm_merged) / 1000
        wall_us = sum(e - s for s, e in union_merged) / 1000
        comm_hidden_us = 0
        for s, e in evts["comm"]:
            comm_hidden_us += overlap_us(s, e, compute_merged) / 1000
        comm_exposed_us = comm_us - comm_hidden_us
        summary["compute_us"] += compute_us
        summary["comm_total_us"] += comm_us
        summary["comm_exposed_us"] += comm_exposed_us
        summary["comm_hidden_us"] += comm_hidden_us
        summary["wall_us"] += wall_us

    n_gpus = len(per_gpu)
    print()
    print(f"{'metric':<20s}  {'avg/GPU (ms)':>14s}  {'sum 8 GPUs (ms)':>17s}")
    print(f"{'-' * 56}")
    print(f"{'compute (busy)':<20s}  {summary['compute_us']/n_gpus/1e3:>14.2f}  {summary['compute_us']/1e3:>17.2f}")
    print(f"{'comm total':<20s}  {summary['comm_total_us']/n_gpus/1e3:>14.2f}  {summary['comm_total_us']/1e3:>17.2f}")
    print(f"  exposed           {summary['comm_exposed_us']/n_gpus/1e3:>14.2f}  {summary['comm_exposed_us']/1e3:>17.2f}")
    print(f"  hidden            {summary['comm_hidden_us']/n_gpus/1e3:>14.2f}  {summary['comm_hidden_us']/1e3:>17.2f}")
    print(f"{'wall (any-kernel)':<20s}  {summary['wall_us']/n_gpus/1e3:>14.2f}  {summary['wall_us']/1e3:>17.2f}")
    if summary["comm_total_us"] > 0:
        pct_exp = summary["comm_exposed_us"] / summary["comm_total_us"] * 100
        print(f"\nNCCL exposed share : {pct_exp:.1f}% of comm time")
    if summary["wall_us"] > 0:
        print(f"NCCL exposed       : {summary['comm_exposed_us']/summary['wall_us']*100:.1f}% of wall")
        print(f"compute            : {summary['compute_us']/summary['wall_us']*100:.1f}% of wall")
    print()
    print(f"Top-{args.top} kernels by total GPU time:")
    print(f"{'%':>5s}  {'cat':<7s}  {'inst':>6s}  {'total_ms':>10s}  name")
    grand_total = sum(cat_total_us.values())
    sorted_kernels = sorted(name_total_us.items(), key=lambda kv: -kv[1])[: args.top]
    for (cat, name), us in sorted_kernels:
        pct = us / grand_total * 100 if grand_total else 0
        # instance count
        inst = sum(1 for s, e, dev, n in rows if n == name and e is not None)
        print(f"{pct:>5.1f}  {cat:<7s}  {inst:>6d}  {us/1e3:>10.2f}  {name[:90]}")


if __name__ == "__main__":
    main()
