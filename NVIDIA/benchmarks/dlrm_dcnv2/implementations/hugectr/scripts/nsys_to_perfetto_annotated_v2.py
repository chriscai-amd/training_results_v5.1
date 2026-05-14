#!/usr/bin/env python3
"""
Convert N consecutive iterations of one GPU's events from an nsys SQLite
export to a Chrome / Perfetto JSON trace, with PyTorch-profiler-style
features:

  1. Per-kernel grid/block/registers/shared-memory info in tooltips (args).
  2. Host launch -> device kernel arrows (via CUPTI correlationId, emitted
     as Chrome flow events; visible in Perfetto when you click a kernel).
  3. Heuristic fwd/bwd pair arrows (per-iter pairing by category).
  4. Model-aware kernel categories ([emb_a2a], [mlp_fwd], [allreduce], ...).

Usage:
    python3 nsys_to_perfetto_annotated_v2.py <input.sqlite> <gpu_id> \
        [first_iter] [n_iters] [output.json]

    python3 nsys_to_perfetto_annotated_v2.py /r/nsys_bs1x_auto.sqlite 0 1000 3 /r/out.json
"""
import sqlite3
import sys
import json
import os
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

DB         = sys.argv[1]
GPU        = int(sys.argv[2])
FIRST_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_ITERS    = int(sys.argv[4]) if len(sys.argv) > 4 else 3
OUT        = sys.argv[5] if len(sys.argv) > 5 else \
             f"{os.path.splitext(DB)[0]}.gpu{GPU}.iter{FIRST_ITER}-{FIRST_ITER + N_ITERS}.v2.json"

con = sqlite3.connect(DB)
cur = con.cursor()

cur.execute("SELECT id, value FROM StringIds")
str_map = {sid: val for sid, val in cur.fetchall()}

# ------------------------------------------------------------------
# Find iteration boundaries via AllReduce kernel
# ------------------------------------------------------------------
ar_ids = [sid for sid, val in str_map.items() if "AllReduce_Sum_f16_RING_LL" in val]
ph = ",".join("?" * len(ar_ids))
cur.execute(f"""
    SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE deviceId = ? AND shortName IN ({ph})
    ORDER BY start
""", [GPU] + ar_ids)
ar_events = cur.fetchall()
print(f"GPU {GPU}: {len(ar_events)} AllReduce events", file=sys.stderr)
if FIRST_ITER + N_ITERS >= len(ar_events):
    print(f"ERROR: only {len(ar_events)} iters", file=sys.stderr); sys.exit(2)

T_START = ar_events[FIRST_ITER - 1][1]
T_END   = ar_events[FIRST_ITER + N_ITERS - 1][1]
print(f"Window: iter {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}  "
      f"T_START={T_START}ns  T_END={T_END}ns  span={(T_END-T_START)/1e6:.2f}ms",
      file=sys.stderr)
iter_marks = [(FIRST_ITER + i, ar_events[FIRST_ITER - 1 + i][1])
              for i in range(N_ITERS + 1)]

# ------------------------------------------------------------------
# Classifier (same as v1, plus a few late additions)
# ------------------------------------------------------------------
def classify(name, demangled=""):
    n = name.lower() if name else ""
    d = demangled.lower() if demangled else ""
    if "ncclsendrecv" in n.replace("_", "") or "sendrecv" in n:
        return ("emb_a2a", "ncclSendRecv")
    if "allreduce" in n:
        return ("allreduce", "ncclAllReduce")
    if any(t in n for t in ["swizzle_keys", "label_and_count", "compress_offset",
                            "split_feat_major", "get_keys_flag",
                            "get_unique_key_same_ev_size", "keys_to_indices",
                            "replicate_bucket_range", "count_keys_per_gpu",
                            "transpose_buckets", "compute_shard_ranges",
                            "concat_keys_and_bucket_range", "mp_cal_src_ptrs",
                            "bucket_range"]):
        return ("sparse_prep", n.split("(")[0].split("::")[-1])
    if "reverse_relu" in n:
        return ("mlp_bwd_dgrad", "reverse_relu")
    if "binaryopkernel" in n:
        return ("fused_fma", "binary_op")
    if "radixsort" in n.replace("_", "") or "radix_sort" in n or "deviceradixsort" in n:
        return ("sparse_prep", "cub::radix_sort")
    if ("scaninit" in n.replace("_", "") or "scankernel" in n.replace("_", "")
            or "devicescaninit" in n or "devicescankernel" in n):
        return ("sparse_prep", "cub::scan")
    if "ragged_static_embedding_table_lookup" in n:
        return ("emb_fwd", "ragged_static_embedding_lookup")
    if "multi_to_one_reduce" in n:
        return ("emb_reduce", "embedding::multi_to_one_reduce")
    if "multi_to_one_warp_per_ev" in n:
        return ("emb_fwd", "embedding::multi_to_one_warp")
    if "one_to_multi_warp_per_ev" in n:
        return ("emb_scatter", "embedding::one_to_multi_warp")
    if "update4_kernel" in n:
        return ("opt_emb", "embedding::adagrad_update4")
    if "ada_grad_update" in n or "adagrad" in n:
        return ("opt_dense", "ada_grad_update")
    if "bgrada" in n or "_bgrad_" in n or "drelu" in n:
        return ("mlp_bwd_dgrad", n.split("(")[0])
    if "bias_relu" in n or "bias_f16_relu" in n:
        return ("mlp_fwd", n.split("(")[0])
    if "concat_fwd" in n:
        return ("interaction", "HugeCTR::concat_fwd")
    if "concat_bwd" in n:
        return ("interaction", "HugeCTR::concat_bwd")
    if "convert_array" in n:
        return ("dtype_cast", "HugeCTR::convert_array")
    if "vector_mul_fma" in n or "vector_fma4" in n or "fma3_align" in n:
        return ("fused_fma", "HugeCTR::vector_fma")
    if "splitkreduce" in n.replace("_", ""):
        return ("mlp_fwd", "cublasLt::splitK_reduce")
    if "cutlass3x_sm100" in n or "nvjet_hsh" in n or "cutlass_80" in n:
        return ("mlp_bwd_wgrad", n.split("(")[0])
    if "binarycrossentropy" in n:
        return ("loss", "BCE")
    if "gemv2t_kernel_val" in n or "gemmk1_kernel" in n:
        return ("mlp_fwd", "cublas::gemv_or_gemmk1")
    if n in ("kernel", "kernel2", "globalkernel"):
        if "cublaslt" in d or "cublas" in d or "gemv" in d or "gemm" in d:
            return ("mlp_fwd", "cublas::generic")
        if "cutlass" in d:
            return ("mlp_bwd_dgrad", "cutlass::generic")
        return ("other", n)
    if "memcpy" in n: return ("memcpy", "memcpy")
    if "memset" in n: return ("memset", "memset")
    return ("other", n.split("(")[0] if name else "?")

COLOR = {
    "sparse_prep":     "olive",
    "emb_fwd":         "good",
    "emb_reduce":      "good",
    "emb_a2a":         "rail_response",
    "emb_scatter":     "good",
    "mlp_fwd":         "rail_animation",
    "interaction":     "rail_idle_busy",
    "mlp_bwd_dgrad":   "rail_load",
    "mlp_bwd_wgrad":   "rail_load",
    "allreduce":       "bad",
    "loss":            "bad",
    "opt_emb":         "yellow",
    "opt_dense":       "yellow",
    "dtype_cast":      "grey",
    "fused_fma":       "grey",
    "memcpy":          "grey",
    "memset":          "grey",
    "other":           "white",
}

PID_GPU = 1
PID_HOST = 0

# ------------------------------------------------------------------
# 1) Load ALL kernel events in window, with grid/block/regs/shmem
# ------------------------------------------------------------------
print("Loading kernels in window (with grid/block/regs/shmem) ...", file=sys.stderr)
cur.execute("""
    SELECT k.start, k.end, k.streamId, k.shortName, k.demangledName,
           k.gridX, k.gridY, k.gridZ,
           k.blockX, k.blockY, k.blockZ,
           k.registersPerThread,
           k.staticSharedMemory, k.dynamicSharedMemory,
           k.correlationId
    FROM CUPTI_ACTIVITY_KIND_KERNEL k
    WHERE k.deviceId = ? AND k.start >= ? AND k.start < ?
    ORDER BY k.start
""", (GPU, T_START, T_END))

events = []
kernel_by_corr = {}     # correlationId -> event index in `events`
cat_count = defaultdict(int)

# also track kernels by category for fwd/bwd pairing later
per_iter_kernels = [[] for _ in range(N_ITERS)]  # list-of-lists

for row in cur.fetchall():
    (start, end, sid, short_id, demangled_id,
     gx, gy, gz, bx, by, bz, regs, sshm, dshm, corr) = row
    short = str_map.get(short_id, "")
    demangled = str_map.get(demangled_id, "")
    cat, role = classify(short, demangled)
    cat_count[cat] += 1

    # which iter does this belong to? (find first AR end > kernel start)
    iter_idx = -1
    for i in range(N_ITERS):
        # iter i runs from ar_events[FIRST_ITER-1+i][1] to ar_events[FIRST_ITER+i][1]
        if ar_events[FIRST_ITER - 1 + i][1] <= start < ar_events[FIRST_ITER + i][1]:
            iter_idx = i
            break

    short_kn = short if len(short) < 64 else short[:61] + "..."
    display = f"[{cat}] {role}" if role and role != short else f"[{cat}] {short_kn}"

    ev_idx = len(events)
    threads_per_block = bx * by * bz
    blocks = gx * gy * gz
    ev = {
        "name": display,
        "cat": cat,
        "ph": "X",
        "ts": (start - T_START) / 1000.0,
        "dur": (end - start) / 1000.0,
        "pid": PID_GPU,
        "tid": sid,
        "args": {
            "kernel": short,
            "demangled": (demangled[:240] + ("..." if len(demangled) > 240 else "")) if demangled else "",
            "iter": (FIRST_ITER + iter_idx) if iter_idx >= 0 else "",
            "duration_us": round((end - start) / 1000.0, 2),
            # PyTorch-profiler-style launch args
            "grid": f"{gx}x{gy}x{gz}",
            "block": f"{bx}x{by}x{bz}",
            "blocks_total": blocks,
            "threads_per_block": threads_per_block,
            "regs_per_thread": regs,
            "smem_static_B":  sshm,
            "smem_dynamic_B": dshm,
            "smem_total_B":   sshm + dshm,
            "correlationId":  corr,
        },
    }
    if cat in COLOR:
        ev["cname"] = COLOR[cat]
    events.append(ev)
    if corr is not None:
        kernel_by_corr[corr] = ev_idx
    if iter_idx >= 0:
        per_iter_kernels[iter_idx].append(ev_idx)

print(f"  kernels: {sum(len(x) for x in per_iter_kernels)} in window  "
      f"(across {N_ITERS} iters)", file=sys.stderr)
print(f"  category breakdown: {dict(cat_count)}", file=sys.stderr)

# ------------------------------------------------------------------
# 2) Memcpy events
# ------------------------------------------------------------------
cur.execute("""
    SELECT start, end, streamId, copyKind
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE deviceId = ? AND start >= ? AND start < ?
    ORDER BY start
""", (GPU, T_START, T_END))
COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "DtoD",
              8: "HtoD_async", 9: "DtoH_async", 10: "DtoD_async"}
for start, end, sid, kind in cur.fetchall():
    events.append({
        "name": f"[memcpy] {COPY_KINDS.get(kind, f'kind_{kind}')}",
        "cat": "memcpy", "cname": "grey",
        "ph": "X",
        "ts": (start - T_START) / 1000.0,
        "dur": (end - start) / 1000.0,
        "pid": PID_GPU, "tid": sid,
    })

# ------------------------------------------------------------------
# 3) Host-side CUDA API events (any thread that launches kernels for our GPU)
# ------------------------------------------------------------------
# We can't reliably identify "the rank that owns GPU X" without context info,
# but we don't need to. Strategy: load ALL host CUDA-API events in the window
# whose correlationId matches one of OUR kernels (i.e., kernels we already
# loaded for this GPU). That way we get the exact host launch for each
# in-window kernel, regardless of which thread issued it.
print("Loading host-side CUDA API events with correlationIds ...", file=sys.stderr)
WANTED_API = ["cudaGraphLaunch", "cudaLaunchKernel", "cudaStreamSynchronize",
              "cudaMemcpyAsync", "cudaEventSynchronize", "cudaDeviceSynchronize",
              "cuLaunchKernelEx"]
wanted_ids = [sid for sid, val in str_map.items()
              if any(val.startswith(w) for w in WANTED_API)]

# Helpful index for the correlationId join (one-time cost ~ a few seconds).
try:
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rt_corr ON CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)")
except sqlite3.OperationalError as e:
    print(f"  (index already exists or DB read-only: {e})", file=sys.stderr)

# Map host thread (globalTid) to a small per-GPU lane id. We learn the
# threads from the actual events so we don't need to hardcode any mapping.
host_tid_to_lane = {}
def lane_of(tid):
    if tid not in host_tid_to_lane:
        host_tid_to_lane[tid] = len(host_tid_to_lane) + 1
    return host_tid_to_lane[tid]

ph_str = ",".join("?" * len(wanted_ids))
host_events_added = 0
launch_pairs = 0

if kernel_by_corr:
    # Pull host events that match our window's kernel correlationIds, in
    # batches (sqlite has a parameter-list limit ~999).
    corr_list = list(kernel_by_corr.keys())
    BATCH = 800
    for i in range(0, len(corr_list), BATCH):
        chunk = corr_list[i:i+BATCH]
        cph = ",".join("?" * len(chunk))
        cur.execute(f"""
            SELECT start, end, nameId, correlationId, globalTid
            FROM CUPTI_ACTIVITY_KIND_RUNTIME
            WHERE correlationId IN ({cph})
        """, chunk)
        for start, end, nid, corr, gtid in cur.fetchall():
            n = str_map.get(nid, f"api_{nid}")
            cname = ("bad"            if "GraphLaunch" in n
                     else "rail_idle_busy" if "Sync"     in n
                     else None)
            tid_lane = lane_of(gtid)
            ev = {
                "name": n.replace("_v10000", "").replace("_v11010", "").replace("_v7000", ""),
                "cat": "host_api",
                "ph": "X",
                "ts": (start - T_START) / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": PID_HOST, "tid": tid_lane,
                "args": {"correlationId": corr, "globalTid": gtid,
                         "duration_us": round((end - start) / 1000.0, 2)},
            }
            if cname:
                ev["cname"] = cname
            events.append(ev)
            host_events_added += 1

            # ---- host->device launch flow arrow ----
            kev = events[kernel_by_corr[corr]]
            host_t1 = (end - T_START) / 1000.0
            events.append({
                "name": "launch", "cat": "launch",
                "ph": "s", "id": corr,
                "ts": host_t1,
                "pid": PID_HOST, "tid": tid_lane,
            })
            events.append({
                "name": "launch", "cat": "launch",
                "ph": "f", "id": corr,
                "ts": kev["ts"],
                "pid": kev["pid"], "tid": kev["tid"],
                "bp": "e",
            })
            launch_pairs += 1

# Also pull cudaStreamSynchronize / cudaMemcpyAsync / cudaGraphLaunch on
# the same host threads we discovered, so the host row is informative
# (these may not have matching kernel correlationIds in the window).
sync_apis = [sid for sid, val in str_map.items()
             if any(val.startswith(t) for t in ("cudaStreamSync", "cudaGraphLaunch",
                                                 "cudaMemcpyAsync", "cudaEventSync",
                                                 "cudaDeviceSync"))]
if host_tid_to_lane and sync_apis:
    tids_known = list(host_tid_to_lane.keys())
    tph = ",".join("?" * len(tids_known))
    aph = ",".join("?" * len(sync_apis))
    cur.execute(f"""
        SELECT start, end, nameId, globalTid, correlationId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE globalTid IN ({tph}) AND nameId IN ({aph})
              AND start >= ? AND start < ?
        ORDER BY start
    """, tids_known + sync_apis + [T_START, T_END])
    for start, end, nid, gtid, corr in cur.fetchall():
        n = str_map.get(nid, f"api_{nid}")
        cname = ("bad"            if "GraphLaunch" in n
                 else "rail_idle_busy" if "Sync"     in n
                 else None)
        ev = {
            "name": n.replace("_v10000", "").replace("_v11010", "").replace("_v7000", ""),
            "cat": "host_api",
            "ph": "X",
            "ts": (start - T_START) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_HOST, "tid": lane_of(gtid),
            "args": {"correlationId": corr, "globalTid": gtid,
                     "duration_us": round((end - start) / 1000.0, 2)},
        }
        if cname:
            ev["cname"] = cname
        events.append(ev)

print(f"  host API events: {host_events_added:,}  "
      f"(launch arrows emitted: {launch_pairs:,})",
      file=sys.stderr)
print(f"  host threads (rank lanes): {len(host_tid_to_lane)}", file=sys.stderr)

# ------------------------------------------------------------------
# 4) Heuristic forward/backward kernel pairing arrows
# ------------------------------------------------------------------
#
# Strategy: per iter, find the BCE loss kernel (single one). Everything
# before BCE is "fwd"; everything after BCE up to AllReduce is "bwd".
# Group fwd/bwd kernels by category. Pair fwd[i] with bwd[len-1-i] within
# the same category (LIFO pairing — last fwd op of layer N has its bwd
# come first).
#
# Only a few categories have meaningful fwd/bwd pairs:
#   mlp_fwd      <-> mlp_bwd_dgrad (or mlp_bwd_wgrad for the weight grad)
#   emb_fwd      <-> emb_scatter   (lookup -> grad scatter)
#   emb_reduce   <-> (the partner is a different reduce; skip)
#   interaction  <-> interaction   (concat_fwd <-> concat_bwd)
#   emb_a2a fwd  <-> emb_a2a bwd   (4 SendRecv per iter; first 2 are fwd, last 2 are bwd)

def kernel_phase(kernel_event, loss_t):
    return "fwd" if kernel_event["ts"] + kernel_event["dur"] <= loss_t else "bwd"

print("Building fwd/bwd pair arrows ...", file=sys.stderr)
PAIRS = [
    ("mlp_fwd",   "mlp_bwd_dgrad"),
    ("interaction", "interaction"),
    ("emb_fwd",   "emb_scatter"),
    ("emb_a2a",   "emb_a2a"),     # within emb_a2a: fwd half pairs with bwd half
]
flow_pair_count = 0
flow_id = 1_000_000_000  # large id to not collide with launch corrIds

for it_i, idxs in enumerate(per_iter_kernels):
    if not idxs:
        continue
    # find loss kernel timestamp
    loss_t = None
    for ix in idxs:
        if events[ix]["cat"] == "loss":
            loss_t = events[ix]["ts"]
            break
    if loss_t is None:
        # fallback: split iter at midpoint
        first_t = events[idxs[0]]["ts"]
        last_t = events[idxs[-1]]["ts"] + events[idxs[-1]]["dur"]
        loss_t = (first_t + last_t) / 2

    # bucket per (cat, phase)
    bucket = defaultdict(list)
    for ix in idxs:
        e = events[ix]
        bucket[(e["cat"], kernel_phase(e, loss_t))].append(ix)

    for cat_fwd, cat_bwd in PAIRS:
        fwds = bucket.get((cat_fwd, "fwd"), [])
        bwds = bucket.get((cat_bwd, "bwd"), [])
        # LIFO pairing: last fwd <-> first bwd
        for i in range(min(len(fwds), len(bwds))):
            fix = fwds[-(i + 1)]
            bix = bwds[i]
            fe, be = events[fix], events[bix]
            flow_id += 1
            cat_label = f"fwd_bwd_{cat_fwd}"
            events.append({
                "name": f"{cat_fwd} -> {cat_bwd} pair",
                "cat": cat_label,
                "ph": "s", "id": flow_id,
                "ts": fe["ts"] + fe["dur"],
                "pid": fe["pid"], "tid": fe["tid"],
            })
            events.append({
                "name": f"{cat_fwd} -> {cat_bwd} pair",
                "cat": cat_label,
                "ph": "f", "id": flow_id,
                "ts": be["ts"],
                "pid": be["pid"], "tid": be["tid"],
                "bp": "e",
            })
            flow_pair_count += 1

print(f"  fwd/bwd flow pairs emitted: {flow_pair_count:,}", file=sys.stderr)

# ------------------------------------------------------------------
# Iter boundary instant markers
# ------------------------------------------------------------------
for label, t in iter_marks:
    events.append({
        "name": f"=== iter {label} (AR end) ===",
        "cat": "iter_marker", "cname": "black",
        "ph": "I", "s": "g",
        "ts": (t - T_START) / 1000.0,
        "pid": PID_GPU, "tid": 0,
    })

# Process / thread metadata
events.append({"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
               "args": {"name": f"GPU {GPU}  (iters {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}, "
                                f"{(T_END-T_START)/1e6:.2f} ms)"}})
events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
               "args": {"name": f"Host (rank {GPU})"}})
unique_streams = sorted({e["tid"] for e in events
                         if e.get("pid") == PID_GPU and e["ph"] == "X"})
for sid in unique_streams:
    events.append({"name": "thread_name", "ph": "M", "pid": PID_GPU, "tid": sid,
                   "args": {"name": f"stream {sid}"}})
# Host thread lane labels
for gtid, lane in host_tid_to_lane.items():
    events.append({"name": "thread_name", "ph": "M", "pid": PID_HOST, "tid": lane,
                   "args": {"name": f"host_thread {gtid}"}})

print(f"\nTotal events: {len(events):,}", file=sys.stderr)
print(f"Writing {OUT} ...", file=sys.stderr)
with open(OUT, "w") as f:
    json.dump({
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "src": DB, "gpu": GPU,
            "iter_window": [FIRST_ITER, FIRST_ITER + N_ITERS - 1],
            "T_START_ns": T_START,
            "window_ms": (T_END - T_START) / 1e6,
            "categories": dict(cat_count),
            "launch_arrows": launch_pairs,
            "fwd_bwd_arrows": flow_pair_count,
        }
    }, f)
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT}  ({sz:.2f} MB)", file=sys.stderr)
con.close()
