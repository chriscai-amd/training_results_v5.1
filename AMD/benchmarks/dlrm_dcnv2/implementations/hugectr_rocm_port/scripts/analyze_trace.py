#!/usr/bin/env python3
"""Bucket per-GPU rocprofv3 kernel trace into compute / RCCL / embedding /
data-prep / fill, compute steady-state per-iter timing, and produce a
comparison table vs NV's published B200 numbers."""
import csv
import re
import sys
from collections import defaultdict

PATH = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/chcai/hugectr_rocm_port/rocprof_out2/trace_kernel_trace.csv"

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

print(f"Reading {PATH} ...", file=sys.stderr)
events = []
with open(PATH) as f:
    rdr = csv.DictReader(f)
    for row in rdr:
        if row["Kind"] != "KERNEL_DISPATCH":
            continue
        events.append({
            "agent":  row["Agent_Id"],
            "stream": int(row["Stream_Id"]),
            "name":   row["Kernel_Name"].strip('"'),
            "start":  int(row["Start_Timestamp"]),
            "end":    int(row["End_Timestamp"]),
        })
print(f"Loaded {len(events):,} kernel events", file=sys.stderr)

by_agent = defaultdict(list)
for e in events:
    by_agent[e["agent"]].append(e)
for a in by_agent: by_agent[a].sort(key=lambda e: e["start"])

agent0 = sorted(by_agent.keys())[0]
ev0 = by_agent[agent0]
rccl0 = [e["start"] for e in ev0 if categorise(e["name"]) == "rccl"]
print(f"Agent {agent0}: {len(ev0):,} kernels, {len(rccl0)} RCCL events",
      file=sys.stderr)

def find_iters(rccl_starts, max_iters=80):
    if not rccl_starts: return []
    gaps = sorted([(rccl_starts[i+1]-rccl_starts[i], i)
                   for i in range(len(rccl_starts)-1)], reverse=True)
    big_gap_idx = sorted([g[1] for g in gaps[:max_iters-1]])
    iters, prev = [], rccl_starts[0]
    for idx in big_gap_idx:
        iters.append((prev, rccl_starts[idx]))
        prev = rccl_starts[idx + 1]
    iters.append((prev, rccl_starts[-1]))
    return iters

iters = find_iters(rccl0, max_iters=50)
print(f"Found {len(iters)} candidate iterations on agent {agent0}", file=sys.stderr)

# Trim warmup + tail
if len(iters) >= 20:
    steady = iters[10:-5]
elif len(iters) > 4:
    steady = iters[2:-2]
else:
    steady = iters
print(f"Using {len(steady)} steady-state iters for analysis", file=sys.stderr)

def busy_in(events_sorted, lo, hi):
    total = 0
    for e in events_sorted:
        if e["end"] < lo: continue
        if e["start"] > hi: break
        total += min(e["end"], hi) - max(e["start"], lo)
    return total

def cat_in(events_sorted, lo, hi):
    cb = defaultdict(int)
    for e in events_sorted:
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
        for c, t in cb.items(): cat_per_a[a][c].append(t)

avg_wall = sum(walls)/len(walls)
print(f"\n=== STEADY-STATE PER-ITER (avg of {len(steady)} iters, avg of {len(by_agent)} GPUs) ===")
print(f"Wall time per iter         : {avg_wall/1e6:.3f} ms")
mean_busy = sum(sum(busy_per_a[a])/len(busy_per_a[a]) for a in by_agent)/len(by_agent)
print(f"GPU busy per iter          : {mean_busy/1e6:.3f} ms ({100*mean_busy/avg_wall:.1f}% of wall)")
print(f"Implied host gap per iter  : {(avg_wall-mean_busy)/1e6:.3f} ms ({100*(avg_wall-mean_busy)/avg_wall:.1f}% of wall)")

# Per-category aggregated
all_cats = set()
for a in by_agent:
    all_cats.update(cat_per_a[a].keys())
cat_avg = {c: sum(sum(cat_per_a[a].get(c, [0]))/len(cat_per_a[a].get(c, [1]))
                  for a in by_agent)/len(by_agent) for c in all_cats}
# Fix: average across agents of (per-agent average per-iter)
cat_avg = {}
for c in all_cats:
    s = 0
    n = 0
    for a in by_agent:
        vs = cat_per_a[a].get(c, [])
        if vs:
            s += sum(vs)/len(vs)
        n += 1
    cat_avg[c] = s / n

print(f"\nPer-category breakdown (sum across streams shows overlap factor):")
print(f"  {'category':<14} {'time_ms':>10} {'%_of_wall':>11}")
for c in sorted(cat_avg, key=lambda x: -cat_avg[x]):
    print(f"  {c:<14} {cat_avg[c]/1e6:>10.3f} {100*cat_avg[c]/avg_wall:>10.1f}%")
total_cat = sum(cat_avg.values())
print(f"  {'TOTAL_KERNEL':<14} {total_cat/1e6:>10.3f} {100*total_cat/avg_wall:>10.1f}%  (>100% = stream overlap)")
print(f"  {'host_gap':<14} {(avg_wall-mean_busy)/1e6:>10.3f} {100*(avg_wall-mean_busy)/avg_wall:>10.1f}%")

# RCCL exposed-time analysis
def merge(intervals):
    if not intervals: return []
    intervals = sorted(intervals)
    out = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= out[-1][1]: out[-1][1] = max(out[-1][1], e)
        else: out.append([s, e])
    return [(a, b) for a, b in out]

def overlap_amount(a_iv, b_iv):
    """Total overlap between intervals in a_iv and b_iv (assume merged)."""
    overlap = 0
    bi = 0
    for s, e in a_iv:
        while bi < len(b_iv) and b_iv[bi][1] <= s: bi += 1
        bj = bi
        while bj < len(b_iv) and b_iv[bj][0] < e:
            overlap += min(b_iv[bj][1], e) - max(b_iv[bj][0], s)
            bj += 1
    return overlap

agent_rccl = {}
agent_rccl_overlapped = {}
for a in by_agent:
    rccl_iv_all = merge([(e["start"], e["end"]) for e in by_agent[a]
                         if categorise(e["name"]) == "rccl"])
    nonrccl_iv_all = merge([(e["start"], e["end"]) for e in by_agent[a]
                            if categorise(e["name"]) != "rccl"])
    total = 0
    overlapped = 0
    for (lo, hi) in steady:
        rccl_iv = [(max(s, lo), min(e, hi)) for s, e in rccl_iv_all
                   if e > lo and s < hi]
        nonrccl_iv = [(max(s, lo), min(e, hi)) for s, e in nonrccl_iv_all
                      if e > lo and s < hi]
        rccl_iv = [(s, e) for s, e in rccl_iv if e > s]
        nonrccl_iv = [(s, e) for s, e in nonrccl_iv if e > s]
        total += sum(e-s for s, e in rccl_iv)
        overlapped += overlap_amount(rccl_iv, nonrccl_iv)
    agent_rccl[a] = total / len(steady)
    agent_rccl_overlapped[a] = overlapped / len(steady)

mean_rccl = sum(agent_rccl.values())/len(agent_rccl)
mean_rccl_ov = sum(agent_rccl_overlapped.values())/len(agent_rccl_overlapped)
print(f"\n=== RCCL OVERLAP (per-iter, avg across {len(by_agent)} GPUs) ===")
print(f"  total RCCL time      : {mean_rccl/1e6:.3f} ms ({100*mean_rccl/avg_wall:.1f}% of wall)")
print(f"  hidden in compute    : {mean_rccl_ov/1e6:.3f} ms "
      f"({100*mean_rccl_ov/mean_rccl if mean_rccl>0 else 0:.1f}% of RCCL)")
print(f"  exposed (compute idle): {(mean_rccl-mean_rccl_ov)/1e6:.3f} ms "
      f"({100*(mean_rccl-mean_rccl_ov)/avg_wall if avg_wall>0 else 0:.1f}% of wall)")

# Top 15 kernels
print(f"\n=== TOP 15 KERNELS (steady, all GPUs) ===")
top = defaultdict(int)
for a, ev in by_agent.items():
    for (lo, hi) in steady:
        for e in ev:
            if e["end"] < lo: continue
            if e["start"] > hi: break
            d = min(e["end"], hi) - max(e["start"], lo)
            if d > 0: top[e["name"]] += d
total_busy = sum(top.values())
print(f"  {'kernel':<70} {'time_ms':>10} {'%':>6} {'cat':<14}")
for name in sorted(top, key=lambda k: -top[k])[:15]:
    short = (name[:67] + "..") if len(name) > 70 else name
    cat = categorise(name)
    print(f"  {short:<70} {top[name]/1e6:>10.2f} {100*top[name]/total_busy:>5.1f}% {cat:<14}")
