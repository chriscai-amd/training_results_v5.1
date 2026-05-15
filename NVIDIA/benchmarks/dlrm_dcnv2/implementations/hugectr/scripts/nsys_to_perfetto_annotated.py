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
    python3 nsys_to_perfetto_annotated.py <input.sqlite> <gpu_id|all> \
        [first_iter] [n_iters] [output.json]

    # single GPU:
    python3 nsys_to_perfetto_annotated.py /r/nsys.sqlite 0 1000 3 /r/out.json

    # all 8 GPUs combined into one Perfetto trace (one process per GPU):
    python3 nsys_to_perfetto_annotated.py /r/nsys.sqlite all 1000 5 /r/out.json
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
GPU_ARG    = sys.argv[2]
FIRST_ITER = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_ITERS    = int(sys.argv[4]) if len(sys.argv) > 4 else 3

ALL_MODE   = (GPU_ARG.lower() == "all")
if not ALL_MODE:
    SINGLE_GPU = int(GPU_ARG)

if len(sys.argv) > 5:
    OUT = sys.argv[5]
else:
    tag = "all" if ALL_MODE else f"gpu{SINGLE_GPU}"
    OUT = f"{os.path.splitext(DB)[0]}.{tag}.iter{FIRST_ITER}-{FIRST_ITER + N_ITERS}.json"

con = sqlite3.connect(DB)
cur = con.cursor()

cur.execute("SELECT id, value FROM StringIds")
str_map = {sid: val for sid, val in cur.fetchall()}

# Discover GPU device IDs present in the trace
cur.execute("SELECT DISTINCT deviceId FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY deviceId")
all_devices = [r[0] for r in cur.fetchall()]
print(f"Devices in trace: {all_devices}", file=sys.stderr)
GPU_LIST = all_devices if ALL_MODE else [SINGLE_GPU]

# ------------------------------------------------------------------
# Find iteration boundaries via AllReduce kernel (use GPU 0 as the
# reference clock; same iter index applies to all GPUs since they all
# step in lockstep at the data-parallel AllReduce).
# ------------------------------------------------------------------
ar_ids = [sid for sid, val in str_map.items() if "AllReduce_Sum_f16_RING_LL" in val]
ph = ",".join("?" * len(ar_ids))
REF_GPU = GPU_LIST[0]
cur.execute(f"""
    SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE deviceId = ? AND shortName IN ({ph})
    ORDER BY start
""", [REF_GPU] + ar_ids)
ar_events = cur.fetchall()
print(f"GPU {REF_GPU} (ref): {len(ar_events)} AllReduce events", file=sys.stderr)
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

def classify_stream(cat_hist, kernel_names, n_memcpy):
    """Assign a semantic role name to a CUDA stream based on the mix of
    kernels that run on it. Mirrors the architecture in
    HugeCTR/include/gpu_resource.hpp + the EBC scheduler's side-stream pool.

    Returns a string label like 'computation_stream', 'wgrad_stream',
    'embedding_mp', 'embedding_dp', 'sparse_prep', 'memcpy_stream', or
    'cublaslt_internal' (caller numbers the latter when there are several).
    """
    n = sum(cat_hist.values())

    # Pure-memcpy lane (no kernels at all -- e.g. the dedicated
    # AsyncDataReader H2D stream).
    if n == 0:
        return "memcpy_stream" if n_memcpy > 0 else "idle"

    def pct(cat):
        return cat_hist.get(cat, 0) / n

    # --- Embedding "mp" lane: carries the embedding all-to-all (ncclSendRecv)
    # together with embedding fwd lookups. In MI350X's table this is the
    # "Embedding 'mp'" stream (emb_fwd + emb_a2a + opt_emb + emb_reduce).
    if pct("emb_a2a") > 0.10 and pct("emb_fwd") > 0.05:
        return "embedding_mp"

    # --- Sparse-prep / embedding-dp lanes: dominated by sparse-prep kernels.
    # In our trace HCTR's EBC scheduler splits these across two side-streams:
    #   * "_a" (sparse_prep): radix-sort + get_unique_key dominant
    #   * "_b" (embedding_dp): scan + keys_to_indices + sometimes secondary
    #     ncclSendRecv -- matches MI350X's "Embedding 'dp'" stream.
    if pct("sparse_prep") > 0.6:
        names_l = " ".join(n.lower() for n in kernel_names)
        radix_score = (names_l.count("radixsort") + names_l.count("radix_sort")
                       + names_l.count("get_unique_key") + names_l.count("get_keys_flag"))
        scan_score  = (names_l.count("devicescan") + names_l.count("scaninit")
                       + names_l.count("keys_to_indices") + names_l.count("compress_offset"))
        # Secondary ncclSendRecv presence means it's the "dp" stream
        if pct("emb_a2a") > 0.0 or scan_score > radix_score:
            return "embedding_dp"
        return "sparse_prep"

    # --- Wgrad lane (computation_stream_2_): identified by the presence of
    # HCTR's Concat-layer BACKWARD kernel WITHOUT the FORWARD counterpart.
    # When async_wgrad=True, HCTR splits the Concat layer such that
    # concat_fwd_kernel runs on computation_stream_ and concat_bwd_kernel
    # runs on computation_stream_2_; that's the unique signature.
    # When async_wgrad=False, both run on computation_stream_ together,
    # which we'll catch in the main-computation rule below.
    names_lower_blob = " ".join(kn.lower() for kn in kernel_names)
    has_concat_fwd = "concat_fwd" in names_lower_blob
    has_concat_bwd = "concat_bwd" in names_lower_blob
    if (has_concat_bwd and not has_concat_fwd
            and pct("emb_a2a") == 0
            and pct("allreduce") == 0):
        return "wgrad_stream"

    # --- Main computation_stream_: large kernel count, broad MLP mix
    mlp_share = (pct("mlp_fwd") + pct("mlp_bwd_dgrad") + pct("mlp_bwd_wgrad")
                 + pct("fused_fma") + pct("interaction"))
    if n >= 100 and mlp_share > 0.30:
        return "computation_stream"

    # --- Otherwise: a cuBLASLt-managed helper stream (split-K reduction lane,
    # batched-GEMM tail, etc.). Caller adds an ordinal suffix.
    return "cublaslt_internal"

# In multi-GPU mode, give each GPU its own process and its host its own
# process. Layout (Perfetto displays processes in PID order):
#   PID 0  ..  N-1   = host (rank 0 .. rank N-1) launches
#   PID 100 .. 100+N = GPU 0 .. GPU N-1 device events
PID_HOST_BASE = 0
PID_GPU_BASE  = 100

# Aggregate stats across GPUs
all_events = []
all_iter_marks_emitted = False
total_kernels = 0
total_host_apis = 0
total_launch_pairs = 0
total_fwdbwd_pairs = 0
gpu_summaries = []

# Pre-create runtime index once (cheap)
try:
    cur.execute("CREATE INDEX IF NOT EXISTS idx_rt_corr ON CUPTI_ACTIVITY_KIND_RUNTIME(correlationId)")
except sqlite3.OperationalError as e:
    print(f"  (index exists or DB read-only: {e})", file=sys.stderr)

# ------------------------------------------------------------------
# Per-GPU processing loop
# ------------------------------------------------------------------
for GPU in GPU_LIST:
    PID_GPU  = PID_GPU_BASE  + GPU
    PID_HOST = PID_HOST_BASE + GPU
    print(f"\n=== Processing GPU {GPU} (PID_GPU={PID_GPU}, PID_HOST={PID_HOST}) ===",
          file=sys.stderr)

    # 1) Kernel events in window for this GPU
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
    kernel_by_corr = {}
    cat_count = defaultdict(int)
    per_iter_kernels = [[] for _ in range(N_ITERS)]
    # per-stream category histograms + kernel-name lists, used to label
    # each stream lane with a semantic role (computation_stream, wgrad_stream,
    # embedding_mp, embedding_dp, sparse_prep, memcpy_stream, cublaslt_internal).
    per_stream_cat_hist = defaultdict(lambda: defaultdict(int))
    per_stream_kernel_names = defaultdict(list)
    per_stream_memcpy_count = defaultdict(int)

    for row in cur.fetchall():
        (start, end, sid, short_id, demangled_id,
         gx, gy, gz, bx, by, bz, regs, sshm, dshm, corr) = row
        short = str_map.get(short_id, "")
        demangled = str_map.get(demangled_id, "")
        cat, role = classify(short, demangled)
        cat_count[cat] += 1
        per_stream_cat_hist[sid][cat] += 1
        per_stream_kernel_names[sid].append(short)

        iter_idx = -1
        for i in range(N_ITERS):
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
                "gpu": GPU,
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

    n_kernels = sum(len(x) for x in per_iter_kernels)
    print(f"  kernels: {n_kernels} in window", file=sys.stderr)
    print(f"  category breakdown: {dict(cat_count)}", file=sys.stderr)

    # 2) Memcpy events
    cur.execute("""
        SELECT start, end, streamId, copyKind
        FROM CUPTI_ACTIVITY_KIND_MEMCPY
        WHERE deviceId = ? AND start >= ? AND start < ?
        ORDER BY start
    """, (GPU, T_START, T_END))
    COPY_KINDS = {1: "HtoD", 2: "DtoH", 3: "DtoD",
                  8: "HtoD_async", 9: "DtoH_async", 10: "DtoD_async"}
    n_memcpy = 0
    for start, end, sid, kind in cur.fetchall():
        events.append({
            "name": f"[memcpy] {COPY_KINDS.get(kind, f'kind_{kind}')}",
            "cat": "memcpy", "cname": "grey",
            "ph": "X",
            "ts": (start - T_START) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_GPU, "tid": sid,
            "args": {"gpu": GPU},
        })
        n_memcpy += 1
        per_stream_memcpy_count[sid] += 1

    # 3) Host-side CUDA API events for this GPU's kernels (correlationId join)
    host_tid_to_lane = {}
    def lane_of(tid):
        if tid not in host_tid_to_lane:
            host_tid_to_lane[tid] = len(host_tid_to_lane) + 1
        return host_tid_to_lane[tid]

    host_events_added = 0
    launch_pairs = 0

    if kernel_by_corr:
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
                             "duration_us": round((end - start) / 1000.0, 2),
                             "gpu": GPU},
                }
                if cname:
                    ev["cname"] = cname
                events.append(ev)
                host_events_added += 1

                kev = events[kernel_by_corr[corr]]
                host_t1 = (end - T_START) / 1000.0
                # Use a globally-unique flow id (GPU<<48 | corr) to avoid
                # collisions with other GPUs' launch ids in the combined trace.
                flow_id = (GPU << 48) | (corr & 0xFFFFFFFF)
                events.append({
                    "name": "launch", "cat": "launch",
                    "ph": "s", "id": flow_id,
                    "ts": host_t1,
                    "pid": PID_HOST, "tid": tid_lane,
                })
                events.append({
                    "name": "launch", "cat": "launch",
                    "ph": "f", "id": flow_id,
                    "ts": kev["ts"],
                    "pid": kev["pid"], "tid": kev["tid"],
                    "bp": "e",
                })
                launch_pairs += 1

    # Also pull cudaStreamSync / cudaGraphLaunch / cudaMemcpyAsync on the
    # same host threads we discovered (useful for showing host idle / waits)
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
                         "duration_us": round((end - start) / 1000.0, 2),
                         "gpu": GPU},
            }
            if cname:
                ev["cname"] = cname
            events.append(ev)

    print(f"  host API events: {host_events_added:,}  "
          f"launch arrows: {launch_pairs:,}  host threads: {len(host_tid_to_lane)}",
          file=sys.stderr)

    # 4) Heuristic forward/backward kernel pairing arrows
    def kernel_phase(kernel_event, loss_t):
        return "fwd" if kernel_event["ts"] + kernel_event["dur"] <= loss_t else "bwd"

    PAIRS = [
        ("mlp_fwd",   "mlp_bwd_dgrad"),
        ("interaction", "interaction"),
        ("emb_fwd",   "emb_scatter"),
        ("emb_a2a",   "emb_a2a"),
    ]
    flow_pair_count = 0
    flow_id_counter = (GPU << 48) | 0x800000000000  # high bit so no collision

    for it_i, idxs in enumerate(per_iter_kernels):
        if not idxs:
            continue
        loss_t = None
        for ix in idxs:
            if events[ix]["cat"] == "loss":
                loss_t = events[ix]["ts"]
                break
        if loss_t is None:
            first_t = events[idxs[0]]["ts"]
            last_t = events[idxs[-1]]["ts"] + events[idxs[-1]]["dur"]
            loss_t = (first_t + last_t) / 2

        bucket = defaultdict(list)
        for ix in idxs:
            e = events[ix]
            bucket[(e["cat"], kernel_phase(e, loss_t))].append(ix)

        for cat_fwd, cat_bwd in PAIRS:
            fwds = bucket.get((cat_fwd, "fwd"), [])
            bwds = bucket.get((cat_bwd, "bwd"), [])
            for i in range(min(len(fwds), len(bwds))):
                fix = fwds[-(i + 1)]
                bix = bwds[i]
                fe, be = events[fix], events[bix]
                flow_id_counter += 1
                cat_label = f"fwd_bwd_{cat_fwd}"
                events.append({
                    "name": f"{cat_fwd} -> {cat_bwd} pair",
                    "cat": cat_label,
                    "ph": "s", "id": flow_id_counter,
                    "ts": fe["ts"] + fe["dur"],
                    "pid": fe["pid"], "tid": fe["tid"],
                })
                events.append({
                    "name": f"{cat_fwd} -> {cat_bwd} pair",
                    "cat": cat_label,
                    "ph": "f", "id": flow_id_counter,
                    "ts": be["ts"],
                    "pid": be["pid"], "tid": be["tid"],
                    "bp": "e",
                })
                flow_pair_count += 1

    print(f"  fwd/bwd pairs: {flow_pair_count:,}", file=sys.stderr)

    # Iter boundary instant markers (only emit once on the lowest GPU,
    # so they don't visually duplicate across rows)
    if not all_iter_marks_emitted:
        for label, t in iter_marks:
            events.append({
                "name": f"=== iter {label} (AR end) ===",
                "cat": "iter_marker", "cname": "black",
                "ph": "I", "s": "g",
                "ts": (t - T_START) / 1000.0,
                "pid": PID_GPU, "tid": 0,
            })
        all_iter_marks_emitted = True

    # Process / thread metadata
    events.append({"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
                   "args": {"name": f"GPU {GPU}  ({n_kernels} kernels in iter "
                                    f"{FIRST_ITER}..{FIRST_ITER+N_ITERS-1})"}})
    events.append({"name": "process_sort_index", "ph": "M", "pid": PID_GPU, "tid": 0,
                   "args": {"sort_index": PID_GPU}})
    events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
                   "args": {"name": f"Host (rank {GPU})"}})
    events.append({"name": "process_sort_index", "ph": "M", "pid": PID_HOST, "tid": 0,
                   "args": {"sort_index": PID_HOST}})
    unique_streams = sorted({e["tid"] for e in events
                             if e.get("pid") == PID_GPU and e["ph"] == "X"})

    # Classify each stream and assign a semantic label. cuBLASLt-internal
    # helpers get numbered suffixes (_1, _2, ...) in busy-time order so the
    # same physical role gets the same suffix across re-runs.
    stream_label = {}
    cublaslt_streams = []  # (busy_ns, sid) -- sorted at end for stable ordering
    for sid in unique_streams:
        hist = per_stream_cat_hist.get(sid, {})
        names = per_stream_kernel_names.get(sid, [])
        n_mc = per_stream_memcpy_count.get(sid, 0)
        role = classify_stream(hist, names, n_mc)
        stream_label[sid] = role
        if role == "cublaslt_internal":
            # measure busy time so we can rank helper streams largest-first
            busy = sum(e["dur"] for e in events
                       if e.get("pid") == PID_GPU and e.get("tid") == sid
                       and e["ph"] == "X")
            cublaslt_streams.append((busy, sid))
    # Number the cuBLASLt-internal helpers
    cublaslt_streams.sort(reverse=True)  # busiest first -> _1
    for i, (_busy, sid) in enumerate(cublaslt_streams, 1):
        stream_label[sid] = f"cublaslt_internal_{i}"

    # Pure-memcpy lanes (streams that only have memcpy events, no kernels)
    memcpy_only_streams = sorted({e["tid"] for e in events
                                  if e.get("pid") == PID_GPU
                                  and e["ph"] == "X" and e.get("cat") == "memcpy"
                                  and e["tid"] not in unique_streams})
    for sid in memcpy_only_streams:
        stream_label[sid] = "memcpy_stream"

    role_summary = defaultdict(int)
    for sid in stream_label:
        role_summary[stream_label[sid]] += 1
    print(f"  stream roles: {dict(role_summary)}", file=sys.stderr)

    for sid in sorted(set(unique_streams) | set(memcpy_only_streams)):
        label = stream_label.get(sid, "unknown")
        events.append({"name": "thread_name", "ph": "M", "pid": PID_GPU, "tid": sid,
                       "args": {"name": f"stream {sid} ({label})"}})
        # also a sort_index so Perfetto orders lanes by role rather than tid
        ROLE_ORDER = {
            "computation_stream":   10,
            "wgrad_stream":         20,
            "embedding_mp":         30,
            "embedding_dp":         40,
            "sparse_prep":          50,
            "memcpy_stream":        60,
            "idle":                 70,
        }
        # cublaslt_internal_* slots after the named roles, in busy-time order
        if label.startswith("cublaslt_internal_"):
            try:
                ord_idx = int(label.rsplit("_", 1)[1])
            except ValueError:
                ord_idx = 0
            sort_key = 100 + ord_idx
        else:
            sort_key = ROLE_ORDER.get(label, 200)
        events.append({"name": "thread_sort_index", "ph": "M",
                       "pid": PID_GPU, "tid": sid,
                       "args": {"sort_index": sort_key}})

    for gtid, lane in host_tid_to_lane.items():
        events.append({"name": "thread_name", "ph": "M", "pid": PID_HOST, "tid": lane,
                       "args": {"name": f"host_thread {gtid}"}})

    all_events.extend(events)
    total_kernels += n_kernels
    total_host_apis += host_events_added
    total_launch_pairs += launch_pairs
    total_fwdbwd_pairs += flow_pair_count
    gpu_summaries.append({
        "gpu": GPU,
        "kernels": n_kernels,
        "memcpys": n_memcpy,
        "host_apis": host_events_added,
        "launch_arrows": launch_pairs,
        "fwd_bwd_arrows": flow_pair_count,
        "categories": dict(cat_count),
        "stream_labels": {str(sid): stream_label[sid]
                          for sid in sorted(stream_label)},
    })

print(f"\n=== Summary ===", file=sys.stderr)
print(f"  GPUs       : {len(GPU_LIST)} {GPU_LIST}", file=sys.stderr)
print(f"  Window     : iter {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}  "
      f"({(T_END-T_START)/1e6:.2f} ms)", file=sys.stderr)
print(f"  Kernels    : {total_kernels:,}", file=sys.stderr)
print(f"  Host APIs  : {total_host_apis:,}", file=sys.stderr)
print(f"  Launch arr : {total_launch_pairs:,}", file=sys.stderr)
print(f"  FwdBwd arr : {total_fwdbwd_pairs:,}", file=sys.stderr)
print(f"  Total events: {len(all_events):,}", file=sys.stderr)

print(f"\nWriting {OUT} ...", file=sys.stderr)
with open(OUT, "w") as f:
    json.dump({
        "traceEvents": all_events,
        "displayTimeUnit": "ms",
        "metadata": {
            "src": DB,
            "gpus": GPU_LIST,
            "iter_window": [FIRST_ITER, FIRST_ITER + N_ITERS - 1],
            "T_START_ns": T_START,
            "window_ms": (T_END - T_START) / 1e6,
            "per_gpu": gpu_summaries,
            "totals": {
                "kernels": total_kernels,
                "host_apis": total_host_apis,
                "launch_arrows": total_launch_pairs,
                "fwd_bwd_arrows": total_fwdbwd_pairs,
            },
        }
    }, f)
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT}  ({sz:.2f} MB)", file=sys.stderr)
con.close()
