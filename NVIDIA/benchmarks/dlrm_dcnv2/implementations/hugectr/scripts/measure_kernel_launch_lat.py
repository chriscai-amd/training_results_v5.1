#!/usr/bin/env python3
"""
Distinguish HOST-side launch latency from GPU-side execution latency.

We measure two quantities for each kernel:
  L_HOST  = (kernel.start - launch_call.end)  ; time after host returned
                                                from cudaLaunchKernel until
                                                kernel actually started on GPU
  L_TOTAL = (kernel.end - launch_call.start)  ; full host->GPU completion

If virtualization is the bottleneck, L_HOST is large (= queue waiting in driver).

We also measure gaps between consecutive kernels on the *same* stream.
"""
import sqlite3
import sys
from collections import defaultdict

DB = sys.argv[1] if len(sys.argv) > 1 else "/r/nsys_validate.sqlite"

con = sqlite3.connect(DB)
cur = con.cursor()

# Find cudaLaunchKernel runtime calls
cur.execute("SELECT id FROM StringIds WHERE value = 'cudaLaunchKernel'")
launch_kernel_id = cur.fetchone()
print(f"cudaLaunchKernel id: {launch_kernel_id}", flush=True)
cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaGraphLaunch%'")
graph_launch_ids = [r[0] for r in cur.fetchall()]
print(f"cudaGraphLaunch ids: {graph_launch_ids}", flush=True)
cur.execute("SELECT id FROM StringIds WHERE value LIKE '%MemcpyAsync%' OR value = 'cudaLaunchKernel'")
launch_ids = [r[0] for r in cur.fetchall()]
print(f"Launch+Memcpy ids: {launch_ids[:5]}", flush=True)

# --- Gap-between-consecutive-kernels on the same stream ---
print("\n=== Inter-kernel gap on same stream (per GPU) ===", flush=True)
print(f"{'GPU':>4} {'stream':>10} {'kernels':>8} {'p50_us':>8} {'p90_us':>8} {'p99_us':>9} {'max_us':>9}")
cur.execute("""
    SELECT deviceId, streamId, start, end
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    ORDER BY deviceId, streamId, start
""")
rows = cur.fetchall()
by_stream = defaultdict(list)
for did, sid, s, e in rows:
    by_stream[(did, sid)].append((s, e))

# For each (GPU, stream) with substantial kernel count, compute gaps
big_streams = []
for (did, sid), evts in by_stream.items():
    if len(evts) >= 500:
        gaps = []
        for i in range(1, len(evts)):
            g = evts[i][0] - evts[i - 1][1]
            if 0 < g < 50_000_000:  # < 50 ms
                gaps.append(g)
        if not gaps:
            continue
        gaps.sort()
        n = len(gaps)
        big_streams.append((did, sid, n, gaps[n // 2], gaps[int(n * 0.9)],
                            gaps[int(n * 0.99)], gaps[-1]))

# Sort by GPU and kernel count
big_streams.sort(key=lambda x: (x[0], -x[2]))
shown = defaultdict(int)
for did, sid, n, p50, p90, p99, mx in big_streams:
    if shown[did] >= 3:
        continue
    shown[did] += 1
    print(f"{did:>4} {sid:>10d} {n:>8d} {p50/1e3:>8.1f} {p90/1e3:>8.1f} {p99/1e3:>9.1f} {mx/1e3:>9.1f}", flush=True)

# --- Per-launch latency: cudaLaunchKernel time vs corresponding kernel start ---
print("\n=== cudaLaunchKernel host-side duration distribution ===", flush=True)
if launch_kernel_id:
    cur.execute("""
        SELECT end - start AS dur
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE nameId = ?
        ORDER BY dur
    """, (launch_kernel_id[0],))
    durs = [r[0] for r in cur.fetchall()]
    if durs:
        n = len(durs)
        durs.sort()
        print(f"n={n:,}", flush=True)
        print(f"  min={durs[0]/1e3:.2f} us, p50={durs[n//2]/1e3:.2f} us, "
              f"p90={durs[int(n*0.9)]/1e3:.2f} us, p99={durs[int(n*0.99)]/1e3:.2f} us, "
              f"max={durs[-1]/1e3:.2f} us")

# --- Compute per-iter accounting ---
print("\n=== Per-iter time decomposition (GPU 0) ===", flush=True)
# trace duration on GPU 0:
gpu0 = [(s, e) for did, sid, s, e in rows if did == 0]
gpu0.sort()
span_start = min(s for s, _ in gpu0)
span_end = max(e for _, e in gpu0)
span = span_end - span_start

# Use one well-defined kernel as iter marker - any kernel that fires 1x/iter
# nccl AllReduce is a good candidate.
cur.execute("SELECT id FROM StringIds WHERE value LIKE '%AllReduce_Sum_f16_RING_LL%' LIMIT 5")
ar_str_ids = [r[0] for r in cur.fetchall()]
print(f"AllReduce string ids: {ar_str_ids}", flush=True)
if ar_str_ids:
    placeholder = ",".join("?" * len(ar_str_ids))
    cur.execute(f"""
        SELECT start, end
        FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = 0 AND shortName IN ({placeholder})
        ORDER BY start
    """, ar_str_ids)
    ar_events = cur.fetchall()
    print(f"GPU 0 AllReduce events: {len(ar_events)}", flush=True)
    if len(ar_events) >= 10:
        # iter cycle = next AR start - this AR start
        cycles = []
        ar_gaps = []  # gap between AR-end and next AR-start
        for i in range(1, len(ar_events)):
            cycle = ar_events[i][0] - ar_events[i - 1][0]
            gap = ar_events[i][0] - ar_events[i - 1][1]
            if 0 < cycle < 100_000_000:
                cycles.append(cycle)
                ar_gaps.append(gap)
        cycles.sort()
        ar_gaps.sort()
        n = len(cycles)
        if n > 0:
            print(f"iter cycle: n={n}, mean={sum(cycles)/n/1e3:.1f} us, "
                  f"p50={cycles[n//2]/1e3:.1f} us, p90={cycles[int(n*0.9)]/1e3:.1f} us")
            print(f"AR-end -> next-AR-start: mean={sum(ar_gaps)/n/1e3:.1f} us, "
                  f"p50={ar_gaps[n//2]/1e3:.1f} us, p90={ar_gaps[int(n*0.9)]/1e3:.1f} us")

# --- Check if there are any non-kernel periods where GPU is idle ---
# Use merged intervals
print("\n=== GPU 0 idle distribution (gaps in merged kernel timeline) ===", flush=True)
intervals = sorted([(s, e) for s, e in gpu0])
merged = []
cs, ce = intervals[0]
for s, e in intervals[1:]:
    if s <= ce:
        ce = max(ce, e)
    else:
        merged.append((cs, ce))
        cs, ce = s, e
merged.append((cs, ce))
gaps = []
for i in range(1, len(merged)):
    g = merged[i][0] - merged[i - 1][1]
    if g > 0:
        gaps.append(g)
gaps.sort()
n = len(gaps)
print(f"  n gaps: {n}, sum: {sum(gaps)/1e6:.2f} ms, "
      f"p50={gaps[n//2]/1e3:.1f} us, "
      f"p90={gaps[int(n*0.9)]/1e3:.1f} us, "
      f"p99={gaps[int(n*0.99)]/1e3:.1f} us, "
      f"max={gaps[-1]/1e3:.1f} us")

# Histogram of idle gaps in 50us buckets
buckets = [0] * 21
for g in gaps:
    b = min(int(g) // 50_000, 20)
    buckets[b] += 1
print("  Gap histogram (us):")
for i, c in enumerate(buckets):
    if c == 0:
        continue
    lo = i * 50
    hi = (i + 1) * 50
    label = f"  [{lo:>4d}-{hi:<4d})" if i < 20 else f"  [>{lo:>4d}    )"
    bar = "#" * min(c // 10, 60)
    print(f"{label}: {c:>5d} {bar}")
con.close()
