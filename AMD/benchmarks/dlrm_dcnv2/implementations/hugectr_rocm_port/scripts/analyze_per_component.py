#!/usr/bin/env python3
"""Per-component breakdown analyzer for rocprofv3 trace, modeled on
NV's b200/README §7.4a + §8.2a methodology:

  - per-iter wall (using RCCL kernels as iter markers on agent 0)
  - GPU busy / idle %
  - per-component (compute / RCCL / embedding / etc.) bucketing
  - RCCL exposed-vs-hidden (overlap with non-RCCL kernels)
  - per-stream inter-kernel-gap p50/p90/p99/max
  - top-N kernels by total time
"""
import csv
import re
import sys
import statistics
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/chcai/hugectr_rocm_port/rocprof_out2/trace_kernel_trace.csv"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "AMD MI350X 8-GPU"

CATEGORIES = [
    ("rccl",         re.compile(r"ncclDevKernel|RCCL|nccl[A-Z]")),
    ("embedding",    re.compile(r"multi_to_one|one_to_multi|one_to_one_atomic|"
                                r"embedding|adagrad|sparse|"
                                r"label_and_count|swizzle|replicate_bucket|"
                                r"compress_offset|index_calc|keys_to_indices|"
                                r"data_distrib|key_filter|dp_index|mp_index|"
                                r"unique_op|hash|bucket_range")),
    ("mlp_gemm",     re.compile(r"^Cijk|hipblaslt|hipBLAS|gemm|GEMM|hgemm|sgemm")),
    ("mlp_elem",     re.compile(r"add_bias|drelu|relu|bias|forward_fc_align|"
                                r"vector_fma|fma_add|matrix_pair_mul|"
                                r"reverse_add_bias|reduce_sum_columns|"
                                r"bgrad_finalize|bgrada_v5|bprop|fprop|"
                                r"add_per_row|matrix_vector_mul|elementwise|"
                                r"convert_array|cast|MultiCross|cross")),
    ("interaction",  re.compile(r"concat|interact|reshape|transpose")),
    ("fill_memcpy",  re.compile(r"fillBuffer|memset|memcpy|Memcpy|copy_with_offset")),
    ("sort_scan",    re.compile(r"cub::|RadixSort|Reduce|Scan|Onesweep|splitK")),
    ("loss_eval",    re.compile(r"loss|cross_entropy|auc|accuracy|sigmoid|BCE")),
]
def categorise(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"

print(f"\n{'='*70}\n=== {LABEL} :: {PATH}\n{'='*70}", file=sys.stderr)
events = []
with open(PATH) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        if row["Kind"] != "KERNEL_DISPATCH": continue
        events.append({
            "agent":  row["Agent_Id"],
            "stream": int(row["Stream_Id"]),
            "name":   row["Kernel_Name"].strip('"'),
            "start":  int(row["Start_Timestamp"]),
            "end":    int(row["End_Timestamp"]),
        })
print(f"Loaded {len(events):,} events", file=sys.stderr)

by_agent = defaultdict(list)
for e in events:
    by_agent[e["agent"]].append(e)
for a in by_agent: by_agent[a].sort(key=lambda e: e["start"])

agent0 = sorted(by_agent.keys())[0]
ev0 = by_agent[agent0]
rccl0 = [e["start"] for e in ev0 if categorise(e["name"]) == "rccl"]

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
if len(iters) >= 30: steady = iters[10:-5]
elif len(iters) > 4: steady = iters[2:-2]
else: steady = iters
print(f"{len(iters)} iters detected, using {len(steady)} steady", file=sys.stderr)

def busy_in(ev_sorted, lo, hi):
    t = 0
    for e in ev_sorted:
        if e["end"] < lo: continue
        if e["start"] > hi: break
        t += min(e["end"], hi) - max(e["start"], lo)
    return t

def cat_in(ev_sorted, lo, hi):
    cb = defaultdict(int)
    for e in ev_sorted:
        if e["end"] < lo: continue
        if e["start"] > hi: break
        d = min(e["end"], hi) - max(e["start"], lo)
        if d > 0: cb[categorise(e["name"])] += d
    return cb

walls = []
busy_per_a = defaultdict(list)
cat_per_a = defaultdict(lambda: defaultdict(list))
for (lo, hi) in steady:
    walls.append(hi - lo)
    for a, ev in by_agent.items():
        busy_per_a[a].append(busy_in(ev, lo, hi))
        cb = cat_in(ev, lo, hi)
        for c, t in cb.items():
            cat_per_a[a][c].append(t)

avg_wall = sum(walls)/len(walls)

print(f"\n--- STEADY-STATE PER-ITER (avg of {len(steady)} iters, {len(by_agent)} GPUs) ---")
print(f"  wall (any-kernel cycle)    {avg_wall/1e6:>8.3f} ms")
mean_busy = sum(sum(busy_per_a[a])/len(busy_per_a[a]) for a in by_agent)/len(by_agent)
gap = avg_wall - mean_busy
print(f"  GPU busy (any kernel)      {mean_busy/1e6:>8.3f} ms ({100*mean_busy/avg_wall:5.1f}%)")
print(f"  GPU idle (host gap)        {gap/1e6:>8.3f} ms ({100*gap/avg_wall:5.1f}%)")

# Per-category aggregate (mean across agents)
all_cats = set()
for a in by_agent:
    all_cats.update(cat_per_a[a].keys())
cat_avg = {}
for c in all_cats:
    s = 0
    for a in by_agent:
        vs = cat_per_a[a].get(c, [])
        s += (sum(vs)/len(vs)) if vs else 0
    cat_avg[c] = s / len(by_agent)

print(f"\n--- PER-CATEGORY (sum > 100% means stream-overlap) ---")
for c in sorted(cat_avg, key=lambda x: -cat_avg[x]):
    print(f"  {c:<14}  {cat_avg[c]/1e6:>8.3f} ms  {100*cat_avg[c]/avg_wall:>5.1f}%")
total_cat = sum(cat_avg.values())
print(f"  {'TOTAL_KERNEL':<14}  {total_cat/1e6:>8.3f} ms  {100*total_cat/avg_wall:>5.1f}%")
print(f"  {'host_gap':<14}  {gap/1e6:>8.3f} ms  {100*gap/avg_wall:>5.1f}%")

# RCCL hidden-vs-exposed
def merge(intervals):
    if not intervals: return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return [(a, b) for a, b in out]

def overlap(a_iv, b_iv):
    if not a_iv or not b_iv: return 0
    o = 0
    bi = 0
    for s, e in a_iv:
        while bi < len(b_iv) and b_iv[bi][1] <= s: bi += 1
        bj = bi
        while bj < len(b_iv) and b_iv[bj][0] < e:
            o += min(b_iv[bj][1], e) - max(b_iv[bj][0], s)
            bj += 1
    return o

agent_rccl, agent_hidden = {}, {}
for a in by_agent:
    rccl_iv_all = merge([(e["start"], e["end"]) for e in by_agent[a]
                         if categorise(e["name"]) == "rccl"])
    nonrccl_iv_all = merge([(e["start"], e["end"]) for e in by_agent[a]
                            if categorise(e["name"]) != "rccl"])
    total = 0
    hidden = 0
    for (lo, hi) in steady:
        ri = [(max(s, lo), min(e, hi)) for s, e in rccl_iv_all if e > lo and s < hi]
        ri = [(s, e) for s, e in ri if e > s]
        ni = [(max(s, lo), min(e, hi)) for s, e in nonrccl_iv_all if e > lo and s < hi]
        ni = [(s, e) for s, e in ni if e > s]
        total += sum(e-s for s, e in ri)
        hidden += overlap(ri, ni)
    agent_rccl[a] = total / len(steady)
    agent_hidden[a] = hidden / len(steady)

mean_rccl = sum(agent_rccl.values())/len(agent_rccl)
mean_hidden = sum(agent_hidden.values())/len(agent_hidden)
exposed = mean_rccl - mean_hidden

print(f"\n--- RCCL OVERLAP (per-iter, avg across {len(by_agent)} GPUs) ---")
print(f"  total RCCL on-GPU      {mean_rccl/1e6:>8.3f} ms ({100*mean_rccl/avg_wall:5.1f}% of wall)")
print(f"  hidden in compute      {mean_hidden/1e6:>8.3f} ms ({100*mean_hidden/mean_rccl if mean_rccl>0 else 0:5.1f}% of RCCL)")
print(f"  exposed (compute idle) {exposed/1e6:>8.3f} ms ({100*exposed/avg_wall:5.1f}% of wall)")

# --- Per-stream inter-kernel-gap distribution (agent 0 only)
print(f"\n--- PER-STREAM GAP DIST (agent 0) ---")
ev0 = by_agent[agent0]
streams_ev = defaultdict(list)
for e in ev0:
    streams_ev[e["stream"]].append(e)
# Restrict to steady window
window_lo = steady[0][0]
window_hi = steady[-1][1]

stream_summary = []
for s, evs in streams_ev.items():
    evs_w = [e for e in evs if e["start"] >= window_lo and e["end"] <= window_hi]
    if len(evs_w) < 50: continue
    evs_w.sort(key=lambda e: e["start"])
    gaps = [evs_w[i+1]["start"] - evs_w[i]["end"] for i in range(len(evs_w)-1)]
    gaps = [g for g in gaps if g > 0]
    if not gaps: continue
    label_top = sorted({categorise(e["name"]) for e in evs_w}, key=lambda c: -sum(ee["end"]-ee["start"] for ee in evs_w if categorise(ee["name"])==c))[:2]
    stream_summary.append({
        "stream": s,
        "kernels": len(evs_w),
        "p50": statistics.median(gaps) / 1000,
        "p90": sorted(gaps)[int(len(gaps)*0.90)] / 1000,
        "p99": sorted(gaps)[int(len(gaps)*0.99)] / 1000,
        "max": max(gaps) / 1000,
        "cats": ','.join(label_top),
    })
stream_summary.sort(key=lambda x: -x["kernels"])
print(f"  {'stream':>6}  {'kernels':>7}  {'p50_us':>8} {'p90_us':>8} {'p99_us':>8} {'max_us':>10}  {'top_cats':<20}")
for s in stream_summary[:8]:
    print(f"  {s['stream']:>6}  {s['kernels']:>7}  {s['p50']:>8.1f} {s['p90']:>8.1f} {s['p99']:>8.1f} {s['max']:>10.1f}  {s['cats']:<20}")

# Per-iter cycle measurement using NCCL allreduce as marker (NV's methodology)
print(f"\n--- PER-ITER CYCLE (RCCL-marker on agent 0) ---")
walls_us = sorted(w/1000 for w in walls)
print(f"  iters detected: {len(walls_us)}")
print(f"  mean         : {sum(walls_us)/len(walls_us):>8.1f} us")
print(f"  p50          : {walls_us[len(walls_us)//2]:>8.1f} us")
print(f"  p90          : {walls_us[int(len(walls_us)*0.90)]:>8.1f} us")
print(f"  p99          : {walls_us[int(len(walls_us)*0.99)]:>8.1f} us")
print(f"  max          : {max(walls_us):>8.1f} us")

# Top 12 kernels in steady window
print(f"\n--- TOP 12 KERNELS (all GPUs, steady window) ---")
top = defaultdict(int)
for a, ev in by_agent.items():
    for (lo, hi) in steady:
        for e in ev:
            if e["end"] < lo: continue
            if e["start"] > hi: break
            d = min(e["end"], hi) - max(e["start"], lo)
            if d > 0: top[e["name"]] += d
total_busy = sum(top.values())
print(f"  {'kernel':<60}  {'time_ms':>8} {'%':>5}  {'cat':<12}")
for name in sorted(top, key=lambda k: -top[k])[:12]:
    short = (name[:57] + "..") if len(name) > 60 else name
    print(f"  {short:<60}  {top[name]/1e6:>8.2f} {100*top[name]/total_busy:>4.1f}%  {categorise(name):<12}")
