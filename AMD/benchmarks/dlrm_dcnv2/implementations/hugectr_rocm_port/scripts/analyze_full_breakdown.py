#!/usr/bin/env python3
"""Full per-component breakdown matching NV's b200/README §8.2c table:
  - iter cycle p50/mean
  - hipGraphLaunch latency p50/p90/p99/max
  - GPU busy fraction (merged across streams)
  - GPU idle per iter
  - Inter-kernel p99 per stream (compute / NCCL / copy)
  - per-call ncclDevKernel duration
  - host const fraction of iter
"""
import csv
import re
import sys
import statistics
from collections import defaultdict

KERNEL_PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/chcai/hugectr_rocm_port/rocprof_bs4x_v2/trace_kernel_trace.csv"
HIP_PATH = sys.argv[2] if len(sys.argv) > 2 else \
    "/home/chcai/hugectr_rocm_port/rocprof_bs4x_v2/trace_hip_api_trace.csv"
LABEL = sys.argv[3] if len(sys.argv) > 3 else "AMD MI350X bs4x"

def is_rccl(name): return 'nccl' in name.lower()
def is_compute(name): return name.startswith('Cijk') or any(k in name for k in [
    'add_bias', 'drelu', 'vector_fma', 'matrix_pair_mul',
    'reduce_sum_columns', 'bgrad', 'add_per_row', 'matrix_vector_mul',
    'convert_array', 'concat', 'cross', 'MultiCross'])
def is_copy(name): return any(k in name for k in [
    'memcpy', 'Memcpy', 'copyBuffer', 'copy_with_offset', 'fillBuffer'])
def is_embedding(name): return any(k in name for k in [
    'multi_to_one', 'one_to_multi', 'embedding', 'label_and_count',
    'swizzle', 'adagrad', 'replicate_bucket', 'compress_offset',
    'index_calc', 'keys_to_indices', 'data_distrib', 'unique_op'])

print(f"\n{'='*70}\n=== {LABEL}\n{'='*70}", file=sys.stderr)
events = []
with open(KERNEL_PATH) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        if row['Kind'] != 'KERNEL_DISPATCH': continue
        events.append({
            'agent':  row['Agent_Id'],
            'stream': int(row['Stream_Id']),
            'name':   row['Kernel_Name'].strip('"'),
            'start':  int(row['Start_Timestamp']),
            'end':    int(row['End_Timestamp']),
        })
print(f"Loaded {len(events):,} kernel events", file=sys.stderr)

by_agent = defaultdict(list)
for e in events:
    by_agent[e['agent']].append(e)
for a in by_agent: by_agent[a].sort(key=lambda e: e['start'])

agent0 = sorted(by_agent.keys())[0]

# --- Iter boundaries via RCCL kernels on agent 0
rccl0 = [e['start'] for e in by_agent[agent0] if is_rccl(e['name'])]
def find_iters(rccl_starts, max_iters=80):
    if not rccl_starts: return []
    gaps = sorted([(rccl_starts[i+1]-rccl_starts[i], i)
                   for i in range(len(rccl_starts)-1)], reverse=True)
    big = sorted([g[1] for g in gaps[:max_iters-1]])
    iters, prev = [], rccl_starts[0]
    for idx in big:
        iters.append((prev, rccl_starts[idx]))
        prev = rccl_starts[idx + 1]
    iters.append((prev, rccl_starts[-1]))
    return iters

iters = find_iters(rccl0, max_iters=70)
steady = iters[10:-5] if len(iters) >= 30 else iters[2:-2]
print(f"{len(iters)} iters, {len(steady)} steady", file=sys.stderr)

# --- Iter cycle distribution
walls = [hi - lo for lo, hi in steady]
walls_us = sorted(w/1000 for w in walls)
def pct(xs, p): return xs[min(len(xs)-1, int(len(xs)*p))]

# --- GPU busy / idle merged across all streams
def busy_in(ev_sorted, lo, hi):
    """Sum of intervals where ANY kernel is running, clipped to [lo,hi]."""
    intervals = [(max(e['start'], lo), min(e['end'], hi))
                 for e in ev_sorted if e['end'] > lo and e['start'] < hi]
    intervals = [(s, e) for s, e in intervals if e > s]
    if not intervals: return 0
    intervals.sort()
    total = 0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    total += cur_e - cur_s
    return total

busy_per_iter = []
for (lo, hi) in steady:
    busy_per_iter.append(busy_in(by_agent[agent0], lo, hi))
mean_busy = sum(busy_per_iter) / len(busy_per_iter)
mean_wall = sum(walls) / len(walls)

# --- Per-stream gap distribution (compute / NCCL / copy)
streams_ev = defaultdict(list)
for e in by_agent[agent0]:
    streams_ev[e['stream']].append(e)
window_lo = steady[0][0]
window_hi = steady[-1][1]

def stream_role(evs):
    cnt = {'rccl': 0, 'compute': 0, 'copy': 0, 'emb': 0, 'other': 0}
    for e in evs:
        if is_rccl(e['name']): cnt['rccl'] += e['end']-e['start']
        elif is_compute(e['name']): cnt['compute'] += e['end']-e['start']
        elif is_copy(e['name']): cnt['copy'] += e['end']-e['start']
        elif is_embedding(e['name']): cnt['emb'] += e['end']-e['start']
        else: cnt['other'] += e['end']-e['start']
    return max(cnt, key=cnt.get)

stream_summary = []
for s, evs in streams_ev.items():
    evs_w = [e for e in evs if e['start'] >= window_lo and e['end'] <= window_hi]
    if len(evs_w) < 50: continue
    evs_w.sort(key=lambda e: e['start'])
    gaps = [evs_w[i+1]['start'] - evs_w[i]['end'] for i in range(len(evs_w)-1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps: continue
    role = stream_role(evs_w)
    stream_summary.append({
        'stream': s, 'role': role, 'kernels': len(evs_w),
        'p50': statistics.median(gaps) / 1000,
        'p90': sorted(gaps)[int(len(gaps)*0.90)] / 1000,
        'p99': sorted(gaps)[int(len(gaps)*0.99)] / 1000,
        'max': max(gaps) / 1000,
    })
stream_summary.sort(key=lambda x: -x['kernels'])

# --- ncclDevKernel per-call duration (steady iters)
nccl_durs = []
for e in by_agent[agent0]:
    if not is_rccl(e['name']): continue
    if e['start'] < window_lo or e['end'] > window_hi: continue
    nccl_durs.append((e['end'] - e['start']) / 1000)
nccl_durs.sort()

# --- HIP API trace (graph launches)
print(f"Loading HIP API trace ...", file=sys.stderr)
hip_events = defaultdict(list)
hip_apis_count = defaultdict(int)
with open(HIP_PATH) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        api = row.get('Function', '').strip('"')
        if not api: continue
        hip_apis_count[api] += 1
        if 'Graph' in api or 'StreamSynchronize' in api or 'EventRecord' in api:
            try:
                dur = int(row['End_Timestamp']) - int(row['Start_Timestamp'])
                hip_events[api].append(dur / 1000)  # us
            except (KeyError, ValueError):
                pass

# --- Output formatted table
print(f"\n=== METRIC SUMMARY (steady-state, agent {agent0}, "
      f"{len(steady)} iters) ===\n")

print(f"{'Metric':<55} {'Value':>20}")
print(f"{'-'*55:<55} {'-'*20:>20}")
print(f"{'iter cycle p50 (RCCL → RCCL marker)':<55} {pct(walls_us, 0.50)/1000:>15.3f} ms")
print(f"{'iter cycle mean':<55} {(sum(walls_us)/len(walls_us))/1000:>15.3f} ms")
print(f"{'iter cycle p90':<55} {pct(walls_us, 0.90)/1000:>15.3f} ms")
print(f"{'iter cycle p99':<55} {pct(walls_us, 0.99)/1000:>15.3f} ms")
print()

# HIP graph launch latencies
for api in sorted(hip_events.keys()):
    durs = sorted(hip_events[api])
    if not durs: continue
    print(f"{api+' p50':<55} {pct(durs, 0.50):>10.2f} us")
    print(f"{api+' p90':<55} {pct(durs, 0.90):>10.2f} us")
    print(f"{api+' p99':<55} {pct(durs, 0.99):>10.2f} us")
    print(f"{api+' max':<55} {max(durs):>10.2f} us")
    print(f"{api+' count':<55} {len(durs):>10}")
print()

print(f"{'GPU busy / iter (merged)':<55} {mean_busy/1e6:>10.3f} ms ({100*mean_busy/mean_wall:5.1f}% of wall)")
print(f"{'GPU idle / iter':<55} {(mean_wall-mean_busy)/1e6:>10.3f} ms ({100*(mean_wall-mean_busy)/mean_wall:5.1f}% of wall)")
print()

# Per-stream gap p99
print(f"{'Stream':<20} {'kernels':>8} {'p50_us':>10} {'p90_us':>10} {'p99_us':>10} {'max_us':>10}")
for s in stream_summary[:6]:
    print(f"  s={s['stream']:<8} ({s['role']:<7})  {s['kernels']:>8} {s['p50']:>10.1f} {s['p90']:>10.1f} {s['p99']:>10.1f} {s['max']:>10.1f}")
print()

if nccl_durs:
    print(f"{'ncclDevKernel per-call p50':<55} {pct(nccl_durs, 0.50):>10.2f} us")
    print(f"{'ncclDevKernel per-call mean':<55} {sum(nccl_durs)/len(nccl_durs):>10.2f} us")
    print(f"{'ncclDevKernel per-call p99':<55} {pct(nccl_durs, 0.99):>10.2f} us")

# host const fraction
if hip_events:
    g_launches = sorted(hip_events.get('hipGraphLaunch', []))
    if g_launches:
        glaunch_p50 = pct(g_launches, 0.50)
        glaunch_mean = sum(g_launches) / len(g_launches)
        print()
        print(f"{'hipGraphLaunch p50':<55} {glaunch_p50:>10.2f} us")
        print(f"{'hipGraphLaunch mean':<55} {glaunch_mean:>10.2f} us")
        print(f"{'hipGraphLaunch count':<55} {len(g_launches):>10}")
        print(f"{'host const fraction (hipGraphLaunch / iter_p50)':<55} "
              f"{100*glaunch_p50/pct(walls_us, 0.50):>9.2f} %")

# Top HIP APIs by count
print(f"\n=== TOP 10 HIP APIs by call count ===")
for api, cnt in sorted(hip_apis_count.items(), key=lambda x: -x[1])[:10]:
    print(f"  {cnt:>8}  {api}")
