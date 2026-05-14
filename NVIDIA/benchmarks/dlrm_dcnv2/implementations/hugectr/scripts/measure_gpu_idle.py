#!/usr/bin/env python3
"""
Measure per-iteration GPU idle gap to validate the virtualization hypothesis.

Approach:
1. Load all CUPTI_ACTIVITY_KIND_KERNEL events from the sqlite db.
2. For each GPU device, build the union of [start, end] intervals across all streams.
3. Compute total wall time spent with NO kernel running on that GPU = "GPU idle".
4. Cross-check by computing per-iteration boundaries from the AllReduce kernel
   that fires once per iter, and measure the gap between consecutive iters'
   AllReduce-end and next-iter's first-kernel-start (this is "exposed host gap").
"""
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "/r/nsys_validate.sqlite"

con = sqlite3.connect(DB)
cur = con.cursor()

# Schema introspection
cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_KERNEL)")
cols = [r[1] for r in cur.fetchall()]
print(f"KERNEL columns: {cols}", flush=True)

cur.execute("PRAGMA table_info(CUPTI_ACTIVITY_KIND_RUNTIME)")
rt_cols = [r[1] for r in cur.fetchall()]
print(f"RUNTIME columns: {rt_cols}", flush=True)

cur.execute("PRAGMA table_info(StringIds)")
print("StringIds cols:", [r[1] for r in cur.fetchall()], flush=True)

# String id lookup for kernel names and API names
cur.execute("SELECT id, value FROM StringIds")
str_map = dict(cur.fetchall())

# Find AllReduce kernel id (1 per iter per GPU)
ar_ids = []
graph_launch_ids = []
for sid, name in str_map.items():
    if "AllReduce_Sum_f16_RING_LL" in name:
        ar_ids.append(sid)
    if "cudaGraphLaunch" in name and "v10000" in name:
        graph_launch_ids.append(sid)
print(f"AllReduce string ids: {ar_ids}", flush=True)
print(f"cudaGraphLaunch string ids: {graph_launch_ids}", flush=True)

# ============================================================
# Part 1: Per-GPU idle-time measurement
# ============================================================
# Schema for TARGET_INFO_GPU
cur.execute("PRAGMA table_info(TARGET_INFO_GPU)")
gpu_cols = [r[1] for r in cur.fetchall()]
print(f"TARGET_INFO_GPU columns: {gpu_cols}", flush=True)

print("\nLoading kernel events ...", flush=True)
cur.execute("""
    SELECT deviceId, streamId, start, end, shortName
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    ORDER BY deviceId, start
""")
rows = cur.fetchall()
print(f"  loaded {len(rows):,} kernel events", flush=True)

per_gpu = defaultdict(list)  # deviceId -> list of (start, end)
per_gpu_per_stream = defaultdict(lambda: defaultdict(list))  # device -> stream -> list
for did, sid, start, end, sname in rows:
    per_gpu[did].append((start, end))
    per_gpu_per_stream[did][sid].append((start, end))

# Trace duration
all_starts = [r[2] for r in rows]
all_ends = [r[3] for r in rows]
t0 = min(all_starts)
t1 = max(all_ends)
trace_ns = t1 - t0
print(f"\nTrace window: {trace_ns/1e9:.3f} s\n", flush=True)

# Per GPU: merge intervals to get true GPU-busy time
print(f"{'GPU':>4} {'busy_ms':>10} {'idle_ms':>10} {'busy%':>7} {'idle%':>7} {'streams':>8}")
total_iter = 5672 // 8
for did in sorted(per_gpu.keys()):
    intervals = sorted(per_gpu[did])
    # Merge overlapping intervals
    merged = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    busy = sum(e - s for s, e in merged)
    # Use the span of THIS gpu's kernels
    span_s = min(s for s, _ in intervals)
    span_e = max(e for _, e in intervals)
    span = span_e - span_s
    idle = span - busy
    print(f"{did:>4} {busy/1e6:>10.2f} {idle/1e6:>10.2f} {100*busy/span:>6.1f}% {100*idle/span:>6.1f}%  "
          f"{len(per_gpu_per_stream[did]):>3d}",
          flush=True)

# ============================================================
# Part 2: Per-iteration boundary analysis
# Use AllReduce as marker (once per iter per GPU)
# Measure: time between AllReduce-end[i] -> first-non-NCCL-kernel-start[i+1]
#         (this is the host gap if the next iter starts late)
# ============================================================
print("\n=== Per-iteration host-gap analysis (using AllReduce as iter marker) ===\n", flush=True)
if ar_ids:
    ar_id = ar_ids[0]
    # AllReduce events per GPU (sorted by start)
    print(f"{'GPU':>4} {'iters':>6} {'p50_gap_us':>12} {'p90_gap_us':>12} {'p99_gap_us':>12} "
          f"{'min':>8} {'max_ms':>9} {'mean_us':>10}")
    for did in sorted(per_gpu.keys()):
        cur.execute(f"""
            SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
            WHERE deviceId = ? AND shortName = ?
            ORDER BY start
        """, (did, ar_id))
        ar_events = cur.fetchall()
        if len(ar_events) < 10:
            continue
        # Gap between consecutive AllReduce ends and next AllReduce start = full iter cycle
        # Better: gap between AR-end[i] and AR-start[i+1]
        gaps = []
        for i in range(1, len(ar_events)):
            gap = ar_events[i][0] - ar_events[i - 1][1]
            if 0 < gap < 100_000_000:  # filter outliers > 100 ms
                gaps.append(gap)
        if not gaps:
            continue
        gaps.sort()
        n = len(gaps)
        p50 = gaps[n // 2]
        p90 = gaps[int(n * 0.9)]
        p99 = gaps[int(n * 0.99)]
        mn = gaps[0]
        mx = gaps[-1]
        mean = sum(gaps) / n
        print(f"{did:>4} {n:>6d} {p50/1e3:>12.1f} {p90/1e3:>12.1f} {p99/1e3:>12.1f} "
              f"{mn/1e3:>8.1f} {mx/1e6:>9.3f} {mean/1e3:>10.1f}", flush=True)
else:
    print("(no AllReduce kernel found)", flush=True)

# ============================================================
# Part 3: cudaGraphLaunch duration histogram
# ============================================================
print("\n=== cudaGraphLaunch duration distribution (host-side, all ranks) ===\n", flush=True)
if graph_launch_ids:
    placeholders = ",".join("?" * len(graph_launch_ids))
    cur.execute(f"""
        SELECT end - start AS dur
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE nameId IN ({placeholders})
        ORDER BY dur
    """, graph_launch_ids)
    durs = [r[0] for r in cur.fetchall()]
    if durs:
        n = len(durs)
        durs.sort()
        print(f"n={n}, min={durs[0]/1e3:.1f} us, "
              f"p50={durs[n//2]/1e3:.1f} us, "
              f"p90={durs[int(n*0.9)]/1e3:.1f} us, "
              f"p99={durs[int(n*0.99)]/1e3:.1f} us, "
              f"max={durs[-1]/1e3:.1f} us")
        # Histogram in 100us buckets up to 2 ms
        buckets = [0] * 25  # 0-100, 100-200, ..., 2400+
        for d in durs:
            b = min(d // 100_000, 24)
            buckets[b] += 1
        print("\n  Histogram (us):")
        for i, c in enumerate(buckets):
            if c == 0:
                continue
            lo = i * 100
            hi = (i + 1) * 100
            label = f"  [{lo:>4d}-{hi:<4d})" if i < 24 else f"  [>{lo:>4d}    )"
            bar = "#" * min(c // 10, 60)
            print(f"{label}: {c:>5d} {bar}")
con.close()
