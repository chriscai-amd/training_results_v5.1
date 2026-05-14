#!/usr/bin/env python3
"""
Convert N consecutive iterations of one GPU's events from an nsys SQLite
export to a Chrome / Perfetto JSON trace, with model-aware annotations.

Annotates each kernel with its DLRM-DCNv2 semantic role:
  - sparse_prep    KJT building (swizzle, label_count, radix sort)
  - emb_fwd        embedding lookup forward (multi_to_one_warp)
  - emb_reduce     embedding multi-hot sum/reduce
  - emb_a2a_fwd    forward all-to-all of embedding outputs (NCCL SendRecv before MLP)
  - mlp_fwd        MLP forward GEMMs (cutlass _bias_relu_aux, nvjet)
  - interaction    concat / cross network
  - mlp_bwd_dgrad  MLP backward data gradient (cutlass _bgrada, _drelu)
  - mlp_bwd_wgrad  MLP backward weight gradient (nvjet bwd shapes)
  - emb_a2a_bwd    backward all-to-all of embedding gradients (NCCL SendRecv after MLP)
  - emb_scatter    embedding gradient scatter (one_to_multi)
  - emb_grad_reduce embedding gradient local reduce
  - allreduce      DDP weight gradient all-reduce
  - opt_emb        embedding optimizer (Adagrad sparse)
  - opt_dense      dense MLP optimizer (Adagrad)
  - dtype_cast     fp32<->fp16 conversion
  - fused_fma      vector_fma helpers
  - memcpy         CUDA memcpy
  - memset         CUDA memset
  - other          uncategorized

Usage:
    python3 nsys_to_perfetto_annotated.py <input.sqlite> <gpu_id> [first_iter] [n_iters] [output.json]

Example:
    python3 nsys_to_perfetto_annotated.py /r/nsys_bs1x_auto.sqlite 0 100 3 /r/out.json
"""
import sqlite3
import sys
import json
import os
import re

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

DB         = sys.argv[1]
GPU        = int(sys.argv[2])
FIRST_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_ITERS    = int(sys.argv[4]) if len(sys.argv) > 4 else 3
OUT        = sys.argv[5] if len(sys.argv) > 5 else \
             f"{os.path.splitext(DB)[0]}.gpu{GPU}.iter{FIRST_ITER}-{FIRST_ITER + N_ITERS}.perfetto.json"

con = sqlite3.connect(DB)
cur = con.cursor()

# String table
cur.execute("SELECT id, value FROM StringIds")
str_map = {sid: val for sid, val in cur.fetchall()}

# ------------------------------------------------------------------
# Find iteration boundaries via AllReduce kernel (1 per iter per GPU)
# ------------------------------------------------------------------
ar_ids = [sid for sid, val in str_map.items() if "AllReduce_Sum_f16_RING_LL" in val]
print(f"AllReduce string ids: {ar_ids}", file=sys.stderr)

ph = ",".join("?" * len(ar_ids))
cur.execute(f"""
    SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE deviceId = ? AND shortName IN ({ph})
    ORDER BY start
""", [GPU] + ar_ids)
ar_events = cur.fetchall()
print(f"GPU {GPU}: {len(ar_events)} AllReduce events (one per iter)", file=sys.stderr)

if FIRST_ITER + N_ITERS >= len(ar_events):
    print(f"ERROR: trace only has {len(ar_events)} iterations, can't grab "
          f"{FIRST_ITER}..{FIRST_ITER + N_ITERS}", file=sys.stderr)
    sys.exit(2)

# Window: from the END of AllReduce[FIRST_ITER-1] to the END of AllReduce[FIRST_ITER+N_ITERS-1]
# (covers exactly N_ITERS iters, beginning right after the prior iter's AR)
T_START = ar_events[FIRST_ITER - 1][1]
T_END   = ar_events[FIRST_ITER + N_ITERS - 1][1]
WINDOW_NS = T_END - T_START
print(f"Window: iter {FIRST_ITER} .. {FIRST_ITER + N_ITERS - 1}", file=sys.stderr)
print(f"  T = [{T_START}, {T_END}] ns  ({WINDOW_NS/1e6:.3f} ms total)", file=sys.stderr)

# Iter boundaries (within window) for marker events
iter_marks = []
for i in range(N_ITERS + 1):
    idx = FIRST_ITER - 1 + i
    iter_marks.append((idx + 1, ar_events[idx][1]))    # iter "label" = idx+1

# ------------------------------------------------------------------
# Kernel-name -> (cat, role) classification
# ------------------------------------------------------------------
def classify(name, demangled=""):
    n = name.lower() if name else ""
    d = demangled.lower() if demangled else ""
    if "ncclsendrecv" in n.replace("_", "") or "sendrecv" in n:
        return ("emb_a2a", "ncclSendRecv")
    if "allreduce" in n:
        return ("allreduce", "ncclAllReduce")
    # ----- sparse prep / KJT building -----
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
    if ("scaninit" in n.replace("_", "")
        or "scankernel" in n.replace("_", "")
        or "devicescaninit" in n
        or "devicescankernel" in n):
        return ("sparse_prep", "cub::scan")
    # ----- embedding forward / lookup -----
    if "ragged_static_embedding_table_lookup" in n:
        return ("emb_fwd", "ragged_static_embedding_lookup")
    if "multi_to_one_reduce" in n:
        return ("emb_reduce", "embedding::multi_to_one_reduce")
    if "multi_to_one_warp_per_ev" in n:
        return ("emb_fwd", "embedding::multi_to_one_warp")
    # ----- embedding backward -----
    if "one_to_multi_warp_per_ev" in n:
        return ("emb_scatter", "embedding::one_to_multi_warp")
    # ----- embedding optimizer -----
    if "update4_kernel" in n:
        # embedding (sparse) Adagrad — present in demangled as embedding::<unnamed>::update4_kernel
        if "embedding" in d or "ragged" in d:
            return ("opt_emb", "embedding::adagrad_update4")
        else:
            return ("opt_emb", "update4_kernel")
    if "ada_grad_update" in n or "adagrad" in n:
        return ("opt_dense", "ada_grad_update")
    # ----- MLP / dense GEMMs -----
    if "bgrada" in n or "_bgrad_" in n:
        return ("mlp_bwd_dgrad", n.split("(")[0])
    if "drelu" in n:
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
        # Generic GEMM not matched above -> probably weight-grad GEMM
        return ("mlp_bwd_wgrad", n.split("(")[0])
    # ----- loss layer -----
    if "binarycrossentropy" in n:
        return ("loss", "BCE")
    if "gemv2t_kernel_val" in n or "gemmk1_kernel" in n:
        return ("mlp_fwd", "cublas::gemv_or_gemmk1 (small)")
    # ----- generic untyped names -----
    if n in ("kernel", "kernel2", "globalkernel"):
        # "kernel"/"kernel2" can appear from cublas/cudnn fallback paths;
        # try to disambiguate via demangled prefix
        if "cublaslt" in d or "cublas" in d or "gemv" in d or "gemm" in d:
            return ("mlp_fwd", "cublas::generic")
        if "cutlass" in d:
            return ("mlp_bwd_dgrad", "cutlass::generic")
        return ("other", n)
    if "memcpy" in n:
        return ("memcpy", "memcpy")
    if "memset" in n:
        return ("memset", "memset")
    return ("other", n.split("(")[0] if name else "?")

# Color mapping for Perfetto (uses Chrome trace "cname")
# Available: good (green), bad (red), yellow, terrible (dark red),
#            grey, white, black, olive, rail_response, etc.
COLOR = {
    "sparse_prep":     "olive",
    "emb_fwd":         "good",          # green: "useful" embedding work
    "emb_reduce":      "good",
    "emb_a2a":         "rail_response", # orange: NCCL alltoall
    "emb_scatter":     "good",
    "emb_grad_reduce": "good",
    "mlp_fwd":         "rail_animation",# blue-ish: compute fwd
    "interaction":     "rail_idle_busy",
    "mlp_bwd_dgrad":   "rail_load",     # purple-ish: bwd data
    "mlp_bwd_wgrad":   "rail_load",
    "allreduce":       "bad",           # red: DDP allreduce (per-iter sync)
    "loss":            "bad",
    "opt_emb":         "yellow",
    "opt_dense":       "yellow",
    "dtype_cast":      "grey",
    "fused_fma":       "grey",
    "memcpy":          "grey",
    "memset":          "grey",
    "other":           "white",
}

# ------------------------------------------------------------------
# Collect kernel events on GPU within window
# ------------------------------------------------------------------
events = []
PID_GPU = 1
PID_HOST = 0

print(f"Loading kernels in window ...", file=sys.stderr)
cur.execute("""
    SELECT start, end, streamId, shortName, demangledName
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE deviceId = ? AND start >= ? AND start < ?
    ORDER BY start
""", (GPU, T_START, T_END))

cat_count = {}
for start, end, sid, short_id, demangled_id in cur.fetchall():
    short = str_map.get(short_id, "")
    demangled = str_map.get(demangled_id, "")
    name = short or demangled or f"kernel_{short_id}"
    cat, role = classify(short, demangled)
    cat_count[cat] = cat_count.get(cat, 0) + 1

    # Display name: short readable role + actual kernel suffix
    short_kn = short if len(short) < 56 else short[:53] + "..."
    display = f"[{cat}] {role}" if role and role != short else f"[{cat}] {short_kn}"

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
            "demangled": demangled[:200] if demangled else "",
            "duration_us": round((end - start) / 1000.0, 2),
        },
    }
    if cat in COLOR:
        ev["cname"] = COLOR[cat]
    events.append(ev)

print(f"  category breakdown: {sorted(cat_count.items(), key=lambda x: -x[1])}",
      file=sys.stderr)

# Memcpy/memset on GPU within window
print(f"Loading memcpys ...", file=sys.stderr)
cur.execute("""
    SELECT start, end, streamId, copyKind
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE deviceId = ? AND start >= ? AND start < ?
    ORDER BY start
""", (GPU, T_START, T_END))
COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "DtoD",
              8: "HtoD_async", 9: "DtoH_async", 10: "DtoD_async"}
for start, end, sid, kind in cur.fetchall():
    kn = COPY_KINDS.get(kind, f"kind_{kind}")
    events.append({
        "name": f"[memcpy] {kn}", "cat": "memcpy", "cname": "grey",
        "ph": "X",
        "ts": (start - T_START) / 1000.0,
        "dur": (end - start) / 1000.0,
        "pid": PID_GPU, "tid": sid,
    })

# Host-side CUDA API on rank's thread (one rank for one GPU)
print(f"Loading host-side CUDA API events ...", file=sys.stderr)
WANTED_API = ["cudaGraphLaunch", "cudaLaunchKernel", "cudaStreamSynchronize",
              "cudaMemcpyAsync", "cudaEventSynchronize", "cudaDeviceSynchronize"]
wanted_ids = [sid for sid, val in str_map.items()
              if any(val.startswith(w) for w in WANTED_API)]
gid_for_graphlaunch = [sid for sid, val in str_map.items()
                       if val == "cudaGraphLaunch_v10000"]
if gid_for_graphlaunch:
    cur.execute("""
        SELECT globalTid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE nameId = ? GROUP BY globalTid ORDER BY 2 DESC LIMIT 8
    """, (gid_for_graphlaunch[0],))
    tids = [r[0] for r in cur.fetchall()]
    pick_tid = tids[GPU] if GPU < len(tids) else tids[0]
    print(f"  host rank tid for GPU {GPU}: {pick_tid}", file=sys.stderr)

    ph_str = ",".join("?" * len(wanted_ids))
    cur.execute(f"""
        SELECT start, end, nameId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE globalTid = ? AND nameId IN ({ph_str})
              AND start >= ? AND start < ?
        ORDER BY start
    """, [pick_tid] + wanted_ids + [T_START, T_END])
    for start, end, nid in cur.fetchall():
        n = str_map.get(nid, f"api_{nid}")
        cname = "bad" if "GraphLaunch" in n else "rail_idle_busy" if "Sync" in n else None
        ev = {
            "name": n.replace("_v10000", "").replace("_v11010", ""),
            "cat": "host_api",
            "ph": "X",
            "ts": (start - T_START) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_HOST, "tid": 1,
        }
        if cname:
            ev["cname"] = cname
        events.append(ev)

# ------------------------------------------------------------------
# Iter boundary markers
# ------------------------------------------------------------------
for label, t in iter_marks:
    events.append({
        "name": f"=== iter {label} boundary (AR end) ===",
        "cat": "iter_marker", "cname": "black",
        "ph": "I", "s": "g",
        "ts": (t - T_START) / 1000.0,
        "pid": PID_GPU, "tid": 0,
    })

# ------------------------------------------------------------------
# Process / thread name metadata
# ------------------------------------------------------------------
events.append({"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
               "args": {"name": f"GPU {GPU}  (iters {FIRST_ITER} .. {FIRST_ITER + N_ITERS - 1}, {WINDOW_NS/1e6:.2f} ms total)"}})
events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
               "args": {"name": f"Host (rank {GPU})"}})

# Stream-id -> friendly name (we observed 9 streams; label by use)
# These IDs differ between traces; we just label them "stream <id>"
unique_streams = sorted({e["tid"] for e in events if e["pid"] == PID_GPU})
for sid in unique_streams:
    events.append({"name": "thread_name", "ph": "M", "pid": PID_GPU, "tid": sid,
                   "args": {"name": f"stream {sid}"}})

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
            "window_ms": WINDOW_NS / 1e6,
            "category_counts": cat_count,
        }
    }, f)
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT}  ({sz:.2f} MB)", file=sys.stderr)
con.close()
