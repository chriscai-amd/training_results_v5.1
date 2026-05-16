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
  5. Stream lane labeling (computation_stream / embedding_mp / embedding_dp /
     sparse_prep / wgrad_stream / memcpy_stream / cublaslt_internal_N).
  6. (NEW) GEMM shape annotation -- if --gemm-catalog is provided, looks
     up each MLP/cross GEMM kernel by tile-shape decoded from its name +
     CUPTI grid dims and attaches exact M/N/K/dtype/op/epilogue/dlrm_layer
     to the kernel's args. Catalog is built once via build_gemm_catalog.py
     on a gemm_init_log.jsonl produced by cublaslt_gemm_logger_shim.so.
  7. (NEW) NCCL NVTX payload ingestion -- if the trace was exported with
     --include-blobs=true, attaches msg_bytes/peer/reduction to each
     ncclSendRecv / ncclAllReduce device kernel by timestamp-matching to
     the host-side NVTX events.

Usage:
    python3 nsys_to_perfetto_annotated.py <input.sqlite> <gpu_id|all> \
        [first_iter] [n_iters] [output.json] [--gemm-catalog=<path>]

    # single GPU:
    python3 nsys_to_perfetto_annotated.py /r/nsys.sqlite 0 1000 3 /r/out.json

    # all 8 GPUs combined into one Perfetto trace (one process per GPU):
    python3 nsys_to_perfetto_annotated.py /r/nsys.sqlite all 1000 5 /r/out.json

    # with GEMM shape annotation:
    python3 nsys_to_perfetto_annotated.py /r/nsys.sqlite all 1000 5 /r/out.json \\
        --gemm-catalog=/r/gemm_catalog.json
"""
import sqlite3
import sys
import json
import os
import re
from collections import defaultdict

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

# Pull out --gemm-catalog=PATH first so positional args index cleanly
gemm_catalog_path = None
posargs = []
for a in sys.argv[1:]:
    if a.startswith("--gemm-catalog="):
        gemm_catalog_path = a.split("=", 1)[1]
    else:
        posargs.append(a)
if len(posargs) < 2:
    print(__doc__); sys.exit(1)

DB         = posargs[0]
GPU_ARG    = posargs[1]
FIRST_ITER = int(posargs[2]) if len(posargs) > 2 else 200
N_ITERS    = int(posargs[3]) if len(posargs) > 3 else 3

ALL_MODE   = (GPU_ARG.lower() == "all")
if not ALL_MODE:
    SINGLE_GPU = int(GPU_ARG)

if len(posargs) > 4:
    OUT = posargs[4]
else:
    tag = "all" if ALL_MODE else f"gpu{SINGLE_GPU}"
    OUT = f"{os.path.splitext(DB)[0]}.{tag}.iter{FIRST_ITER}-{FIRST_ITER + N_ITERS}.json"

# ----------------------------------------------------------------------
# Load optional GEMM catalog (one-shot output of build_gemm_catalog.py).
# When present, every MLP/cross GEMM kernel in the trace gets exact
# M/N/K/dtype/op/epilogue/dlrm_layer attached to its args.
#
# Two matching strategies are tried per kernel:
#   A. TILE-decode the kernel name -> compute (M=grid_x*tile_M, N=grid_y*tile_N)
#      and look up in CATALOG_BY_MN. Works for the "clean" cutlass3x_sm100_*
#      kernel naming convention. Fails for cutlass *_2sm / *_bgrada variants
#      and for the entire nvjet_hsh_* family (their tile-name-to-grid
#      relation is non-linear / not publicly documented).
#   B. SEQUENCE-based fallback: HCTR's captured CUDA graph replays the same
#      GEMM kernel sequence per iter in a deterministic order. Build a
#      "per-iter template" of (M,N,K,op_a,op_b,epi) tuples from the shim's
#      JSONL log (which captures the exact host-side call order during
#      graph capture). Then for each iter window in the trace, the Nth
#      main-GEMM kernel on the compute stream pairs with the Nth entry
#      in the template. This catches everything the tile decoder misses.
# ----------------------------------------------------------------------
GEMM_CATALOG = None
CATALOG_BY_MN = {}
CATALOG_BY_SHAPE = {}  # (M,N,K,op_a,op_b,epi) -> full record
ITER_TEMPLATE  = []    # list of (M,N,K,op_a,op_b,epi) tuples, one entry per
                       # cublasLtMatmul call within a single HCTR iter

def _build_iter_template(jsonl_path):
    """Read the shim's gemm_init_log.jsonl and infer the per-iter call
    sequence. HCTR's graph capture happens once at the start of training:
    every cublasLtMatmul host call fires *during* that capture, then is
    baked into the captured graph and never fires again (graph replay is
    pure device-side). The shim records one pass per HCTR worker thread
    (one thread per local GPU = 8 threads in our 1x8 setup); each thread's
    call sequence IS the per-iter sequence.

    We pick the thread with the most calls (most complete sequence) and
    return its (shape) tuple list in ts_ns order."""
    calls = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                calls.append(json.loads(line))
    except FileNotFoundError:
        return []
    if not calls:
        return []
    by_tid = defaultdict(list)
    for c in calls:
        by_tid[c["tid"]].append(c)
    # All threads run the same model graph, so any thread works. Pick the
    # one with the most calls (in case any threads logged less due to
    # graph capture happening mid-thread).
    busiest_tid = max(by_tid.keys(), key=lambda t: len(by_tid[t]))
    seq = sorted(by_tid[busiest_tid], key=lambda c: c["ts_ns"])
    return [(c["M"], c["N"], c["K"], c["op_a"], c["op_b"], c["epilogue"])
            for c in seq]

if gemm_catalog_path:
    with open(gemm_catalog_path) as f:
        GEMM_CATALOG = json.load(f)
    CATALOG_BY_MN = GEMM_CATALOG.get("by_mn", {})
    for r in GEMM_CATALOG.get("records", []):
        for epi in r.get("epilogues", ["DEFAULT"]):
            CATALOG_BY_SHAPE[(r["M"], r["N"], r["K"], r["op_a"],
                              r["op_b"], epi)] = r
    src = GEMM_CATALOG.get("source_jsonl", "")
    # Resolve src robustly: absolute, or relative to catalog file's directory
    if src and not os.path.isabs(src):
        candidate = os.path.join(os.path.dirname(os.path.abspath(gemm_catalog_path)), src)
        if os.path.exists(candidate):
            src = candidate
    if src and os.path.exists(src):
        ITER_TEMPLATE = _build_iter_template(src)
    elif src:
        print(f"  WARNING: source_jsonl '{src}' not found; sequence matching disabled",
              file=sys.stderr)
    print(f"Loaded GEMM catalog: {GEMM_CATALOG['n_records']} unique shapes  "
          f"({len(CATALOG_BY_SHAPE)} (shape,epi) keys)  from {src}",
          file=sys.stderr)
    print(f"  iter template: {len(ITER_TEMPLATE)} GEMM calls per iter  "
          f"(sequence-based fallback {'enabled' if ITER_TEMPLATE else 'DISABLED'})",
          file=sys.stderr)

def is_main_gemm_kernel(name):
    """Identify kernels that are "the main GEMM kernel" of one
    cublasLtMatmul call -- as opposed to tail kernels (splitKreduce)
    that some matmul algos launch in addition. cuBLAS-Lt emits exactly
    one of these per matmul call, so they pair 1:1 with the shim log's
    JSONL entries."""
    if not name:
        return False
    return (name.startswith("cutlass3x_") or
            name.startswith("nvjet_hsh_") or
            name == "gemmk1_kernel" or
            name == "Kernel2")

def cat_from_layer(layer_name):
    """Derive the canonical kernel category from a dlrm_layer label
    (set by the catalog). The kernel-name-based classify() heuristic
    flags ALL nvjet_hsh / cutlass3x kernels as 'mlp_bwd_wgrad' (it can't
    tell which from the name alone), but the catalog knows the actual
    role from the matmul's (M, N, K, ops, epi) signature. This lets us
    override the wrong-but-syntactically-correct guess.

    Maps:
      *_fwd                       -> mlp_fwd   (or 'interaction' for cross_*)
      *_bwd_dgrad                 -> mlp_bwd_dgrad
      *_bwd_wgrad                 -> mlp_bwd_wgrad
    """
    if not layer_name:
        return None
    is_cross = layer_name.startswith("cross_")
    if layer_name.endswith("_bwd_dgrad"):
        return "mlp_bwd_dgrad"
    if layer_name.endswith("_bwd_wgrad"):
        return "mlp_bwd_wgrad"
    if layer_name.endswith("_fwd"):
        return "interaction" if is_cross else "mlp_fwd"
    return None

# Tile-shape parsers for the kernel families HCTR uses on B200.
#   nvjet_hsh_<TileM>x<TileN>_<WarpM>x<WarpN>_<...>_h_<flags>_<ops>
#     - example: "nvjet_hsh_448x128_64x2_1x2_h_bz_bias_NNT"
#       -> tile_m=448 tile_n=128 ops=NN_T  (last 3 chars before suffix)
#   cutlass3x_sm100_tensorop_s<TileM>x<TileN>x<TileK>gemm_<flags>_<dtypes>
#     - example: "cutlass3x_sm100_tensorop_s128x256x16gemm_bgrada_f16_..."
#       -> tile_m=128 tile_n=256 tile_k=16
RE_NVJET    = re.compile(r"^nvjet_hsh_(\d+)x(\d+)_(\d+)x(\d+)_.*?_([NT][NT][NT])(?:$|_)")
RE_NVJET_ANY = re.compile(r"^nvjet_hsh_(\d+)x(\d+)_")
RE_CUTLASS  = re.compile(r"^cutlass3x_sm\d+_tensorop_s(\d+)x(\d+)x(\d+)gemm_")
RE_KERNEL2  = re.compile(r"^(?:Kernel2|gemmk1_kernel|splitKreduce_kernel|globalKernel)")  # GEMM tail kernels

def decode_gemm_tile(short_name, grid_x, grid_y, grid_z):
    """Return (tile_m, tile_n, tile_k_or_None, ops_or_None) decoded from
    kernel name; or None if not a recognized GEMM kernel."""
    if not short_name:
        return None
    m = RE_CUTLASS.match(short_name)
    if m:
        tm, tn, tk = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (tm, tn, tk, None)
    m = RE_NVJET.match(short_name)
    if m:
        tm, tn = int(m.group(1)), int(m.group(2))
        ops = m.group(5)  # "NNT" -> (op_a, op_b, layout_C)
        return (tm, tn, None, ops)
    m = RE_NVJET_ANY.match(short_name)
    if m:
        return (int(m.group(1)), int(m.group(2)), None, None)
    return None

def lookup_gemm_shape(short_name, grid_x, grid_y):
    """Given a kernel name + CUPTI grid, return the matching catalog
    record or None. Strategy: decode tile -> compute candidate (M,N) ->
    look up in CATALOG_BY_MN. If multiple candidates (e.g. same M,N with
    different K because of fwd vs bwd_wgrad/dgrad), use the OPS string
    from the kernel name to disambiguate when possible."""
    if not CATALOG_BY_MN:
        return None
    decoded = decode_gemm_tile(short_name, grid_x, grid_y, 1)
    if not decoded:
        return None
    tm, tn, tk, ops = decoded
    # Candidate (M, N) -- output is M x N, M=grid_x*tile_m, N=grid_y*tile_n.
    cand_M = grid_x * tm
    cand_N = grid_y * tn
    candidates = CATALOG_BY_MN.get(f"{cand_M}x{cand_N}", [])
    if not candidates:
        # Some cutlass / nvjet variants swap M/N convention -- try transposed.
        candidates = CATALOG_BY_MN.get(f"{cand_N}x{cand_M}", [])
        if candidates:
            cand_M, cand_N = cand_N, cand_M
    if not candidates:
        return None
    # Disambiguate by ops if we know them. The nvjet "NNT" suffix is
    # (op_a, op_b, output_layout); first two chars are the cublas op_a/op_b.
    if ops and len(ops) >= 2 and len(candidates) > 1:
        op_pair = ops[:2]
        filtered = [r for r in candidates if (r["op_a"] + r["op_b"]) == op_pair]
        if filtered:
            candidates = filtered
    # Pick the largest-K entry (most common case: only one entry per M,N,ops)
    return max(candidates, key=lambda r: r["K"])

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

        # GEMM shape annotation -- only for kernels classified as
        # MLP-family (fwd/bwd_dgrad/bwd_wgrad) or interaction (concat).
        gemm_shape = None
        if cat in ("mlp_fwd", "mlp_bwd_dgrad", "mlp_bwd_wgrad", "interaction"):
            gemm_shape = lookup_gemm_shape(short, gx, gy)
            if gemm_shape and gemm_shape.get("dlrm_layer"):
                # Upgrade the category from the layer hint (since name-based
                # classify() flags all cutlass3x/nvjet_hsh as mlp_bwd_wgrad)
                new_cat = cat_from_layer(gemm_shape["dlrm_layer"])
                if new_cat:
                    cat = new_cat
                # promote the dlrm-layer name into the display string so
                # Perfetto's compact kernel labels are immediately readable
                display = f"[{cat}] {gemm_shape['dlrm_layer']} ({short_kn})"

        ev_idx = len(events)
        threads_per_block = bx * by * bz
        blocks = gx * gy * gz
        args = {
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
        }
        if gemm_shape:
            args["gemm_M"]         = gemm_shape["M"]
            args["gemm_N"]         = gemm_shape["N"]
            args["gemm_K"]         = gemm_shape["K"]
            args["gemm_op_A"]      = gemm_shape["op_a"]
            args["gemm_op_B"]      = gemm_shape["op_b"]
            args["gemm_dt_A"]      = gemm_shape["dt_a"]
            args["gemm_dt_B"]      = gemm_shape["dt_b"]
            args["gemm_dt_C"]      = gemm_shape["dt_c"]
            args["gemm_compute"]   = gemm_shape["compute"]
            args["gemm_epilogue"]  = ",".join(gemm_shape["epilogues"])
            args["gemm_gflops"]    = gemm_shape["gflops"]
            args["gemm_bytes"]     = gemm_shape["bytes_accessed"]
            args["dlrm_layer"]     = gemm_shape.get("dlrm_layer", "")
            args["gemm_match"]     = "tile"
        ev = {
            "name": display,
            "cat": cat,
            "ph": "X",
            "ts": (start - T_START) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_GPU,
            "tid": sid,
            "args": args,
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

    # ---- Sequence-based GEMM annotation (catches kernels the tile decoder missed) ----
    seq_annotated = 0
    seq_skipped = 0
    if ITER_TEMPLATE and CATALOG_BY_SHAPE:
        for it_i, idxs in enumerate(per_iter_kernels):
            # Filter to main GEMM kernels in chronological order
            gemm_in_iter = [ix for ix in idxs
                            if is_main_gemm_kernel(events[ix]["args"].get("kernel", ""))]
            # gemm_in_iter is already in start-time order because per_iter_kernels
            # was built by iterating the cursor's ORDER BY start.
            for pos, ev_idx in enumerate(gemm_in_iter):
                if pos >= len(ITER_TEMPLATE):
                    break  # more trace kernels than template entries -- bail
                if "gemm_M" in events[ev_idx]["args"]:
                    continue  # tile decoder already annotated this one
                shape_key = ITER_TEMPLATE[pos]
                cat_entry = CATALOG_BY_SHAPE.get(shape_key)
                if not cat_entry:
                    seq_skipped += 1
                    continue
                a = events[ev_idx]["args"]
                a["gemm_M"]         = cat_entry["M"]
                a["gemm_N"]         = cat_entry["N"]
                a["gemm_K"]         = cat_entry["K"]
                a["gemm_op_A"]      = cat_entry["op_a"]
                a["gemm_op_B"]      = cat_entry["op_b"]
                a["gemm_dt_A"]      = cat_entry["dt_a"]
                a["gemm_dt_B"]      = cat_entry["dt_b"]
                a["gemm_dt_C"]      = cat_entry["dt_c"]
                a["gemm_compute"]   = cat_entry["compute"]
                a["gemm_epilogue"]  = ",".join(cat_entry["epilogues"])
                a["gemm_gflops"]    = cat_entry["gflops"]
                a["gemm_bytes"]     = cat_entry["bytes_accessed"]
                a["dlrm_layer"]     = cat_entry.get("dlrm_layer", "")
                a["gemm_match"]     = "sequence"
                # Upgrade the kernel's category from the layer hint, AND
                # update the display name + Perfetto color. The original
                # classify() heuristic flags every nvjet_hsh kernel as
                # 'mlp_bwd_wgrad' because the name alone is ambiguous; the
                # catalog knows the actual role.
                if cat_entry.get("dlrm_layer"):
                    new_cat = cat_from_layer(cat_entry["dlrm_layer"])
                    if new_cat:
                        events[ev_idx]["cat"] = new_cat
                        if new_cat in COLOR:
                            events[ev_idx]["cname"] = COLOR[new_cat]
                    kn = a["kernel"]
                    kn_short = kn if len(kn) < 56 else kn[:53] + "..."
                    cat_str = events[ev_idx]["cat"]
                    events[ev_idx]["name"] = f"[{cat_str}] {cat_entry['dlrm_layer']} ({kn_short})"
                seq_annotated += 1
    print(f"  GEMM annotation: tile-matched {sum(1 for e in events if e['ph']=='X' and e.get('args',{}).get('gemm_match','tile')=='tile' and 'gemm_M' in e.get('args',{}))} + "
          f"seq-matched {seq_annotated} (catalog miss: {seq_skipped})",
          file=sys.stderr)

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

    # 3b) NCCL NVTX payload ingestion -- pull host-side ncclSend/Recv/AllReduce
    # NVTX events in the iter window (requires the trace to have been
    # exported with `nsys export --include-blobs=true`). Each event has
    # a jsonText payload like:
    #   {"NCCL communicator ID":..., "Message size [bytes]":..., "Peer rank":3}
    # We render them as X events on a dedicated "host_nccl" lane in the
    # host process; the args carry the full payload, visible in Perfetto
    # tooltips. NCCL kernels on the device side are NOT directly tagged
    # (matching them 1:1 is fragile due to ncclGroupStart/End batching),
    # but their per-iter aggregate is visible in this new lane.
    nccl_lane = 90  # well above the typical host_thread lane ids (1..8)
    host_tid_to_lane.setdefault(("nccl_synth", GPU), nccl_lane)
    nccl_text_ids = [sid for sid, val in str_map.items()
                     if val in ('ncclSend', 'ncclRecv', 'ncclAllReduce',
                                'ncclBroadcast', 'ncclReduce', 'ncclReduceScatter',
                                'ncclAllGather', 'ncclGroupStart', 'ncclGroupEnd')]
    n_nccl_emitted = 0
    nccl_total_bytes = 0
    nccl_op_counts = defaultdict(int)
    if nccl_text_ids:
        nph = ",".join("?" * len(nccl_text_ids))
        cur.execute(f"""
            SELECT start, end, textId, jsonText
            FROM NVTX_EVENTS
            WHERE textId IN ({nph}) AND start >= ? AND start < ?
            ORDER BY start
        """, nccl_text_ids + [T_START, T_END])
        for start, end, tid, jt in cur.fetchall():
            op_name = str_map.get(tid, f"nccl_{tid}")
            payload = {}
            if jt:
                try:
                    payload = json.loads(jt)
                except json.JSONDecodeError:
                    payload = {"raw_jsonText": jt}
            msg_bytes = payload.get("Message size [bytes]")
            peer      = payload.get("Peer rank")
            redop     = payload.get("Reduction operation")
            comm_id   = payload.get("NCCL communicator ID")
            # Build a compact display name: "ncclSend bytes=82944 peer=3"
            bits = [op_name]
            if msg_bytes is not None:
                bits.append(f"bytes={msg_bytes}")
            if peer is not None:
                bits.append(f"peer={peer}")
            if redop is not None:
                bits.append(f"op={redop}")
            ev_args = {
                "op": op_name,
                "duration_us": round((end - start) / 1000.0, 2),
                "gpu": GPU,
            }
            if msg_bytes is not None: ev_args["msg_bytes"] = msg_bytes
            if peer is not None:      ev_args["peer_rank"] = peer
            if redop is not None:     ev_args["reduction"] = redop
            if comm_id is not None:   ev_args["comm_id"]   = str(comm_id)
            events.append({
                "name": " ".join(bits),
                "cat":  "nccl_nvtx",
                "cname": ("bad" if "AllReduce" in op_name else "rail_response"),
                "ph": "X",
                "ts": (start - T_START) / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": PID_HOST,
                "tid": nccl_lane,
                "args": ev_args,
            })
            n_nccl_emitted += 1
            if msg_bytes is not None:
                nccl_total_bytes += msg_bytes
                nccl_op_counts[op_name] += 1
    print(f"  NCCL NVTX events: {n_nccl_emitted:,} emitted  "
          f"(total payload: {nccl_total_bytes/1e6:.1f} MB, "
          f"by op: {dict(nccl_op_counts)})", file=sys.stderr)

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
        if gtid == ("nccl_synth", GPU):
            name = "host_nccl (synth from NVTX payload)"
        else:
            name = f"host_thread {gtid}"
        events.append({"name": "thread_name", "ph": "M", "pid": PID_HOST, "tid": lane,
                       "args": {"name": name}})

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
        "nccl_nvtx_events":  n_nccl_emitted,
        "nccl_total_bytes":  nccl_total_bytes,
        "nccl_op_counts":    dict(nccl_op_counts),
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
