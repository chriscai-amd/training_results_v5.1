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

# Pull out --gemm-catalog=PATH and --py-trace=PATH first so positional args index cleanly
gemm_catalog_path = None
py_trace_path = None
posargs = []
for a in sys.argv[1:]:
    if a.startswith("--gemm-catalog="):
        gemm_catalog_path = a.split("=", 1)[1]
    elif a.startswith("--py-trace="):
        py_trace_path = a.split("=", 1)[1]
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
ITER_TEMPLATE_RICH = [] # list of dicts (one per template entry) carrying
                        # the per-call src_function + stack_hctr from the
                        # shim's JSONL log. Same length as ITER_TEMPLATE.

def _build_iter_template(jsonl_path):
    """Read the shim's gemm_init_log.jsonl and infer the per-iter call
    sequence. HCTR's graph capture happens once at the start of training:
    every cublasLtMatmul host call fires *during* that capture, then is
    baked into the captured graph and never fires again (graph replay is
    pure device-side). The shim records one pass per HCTR worker thread
    (one thread per local GPU = 8 threads in our 1x8 setup); each thread's
    call sequence IS the per-iter sequence.

    Returns (shape_tuples, rich_records) -- both same length, ts_ns
    ordered. shape_tuples is the lookup key for CATALOG_BY_SHAPE; rich
    records carry per-call src_function + HCTR-only stack frames."""
    calls = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                calls.append(json.loads(line))
    except FileNotFoundError:
        return [], []
    if not calls:
        return [], []
    by_tid = defaultdict(list)
    for c in calls:
        by_tid[c["tid"]].append(c)
    # All threads run the same model graph; pick the busiest one.
    busiest_tid = max(by_tid.keys(), key=lambda t: len(by_tid[t]))
    seq = sorted(by_tid[busiest_tid], key=lambda c: c["ts_ns"])
    shape_keys = [(c["M"], c["N"], c["K"], c["op_a"], c["op_b"], c["epilogue"])
                  for c in seq]
    rich = []
    for c in seq:
        stack = c.get("stack", [])
        # Extract HCTR-only frames, drop arg-list parens for compactness
        stack_hctr = [f.split("(")[0][:120] for f in stack if "HugeCTR" in f][:6]
        # Pick best representative function (same heuristic as catalog builder)
        src_function = ""
        for needle in ("MLPLayer", "MultiCrossLayer", "FusedReluBiasFully",
                       "FullyConnectedLayer", "FusedFCLayerFunctors",
                       "GemmFunctor"):
            for f in stack:
                if needle in f:
                    src_function = f.split("(")[0][:90]
                    break
            if src_function:
                break
        rich.append({"src_function": src_function, "stack_hctr": stack_hctr})
    return shape_keys, rich

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
        ITER_TEMPLATE, ITER_TEMPLATE_RICH = _build_iter_template(src)
    elif src:
        print(f"  WARNING: source_jsonl '{src}' not found; sequence matching disabled",
              file=sys.stderr)
    print(f"Loaded GEMM catalog: {GEMM_CATALOG['n_records']} unique shapes  "
          f"({len(CATALOG_BY_SHAPE)} (shape,epi) keys)  from {src}",
          file=sys.stderr)
    print(f"  iter template: {len(ITER_TEMPLATE)} GEMM calls per iter  "
          f"(sequence-based fallback {'enabled' if ITER_TEMPLATE else 'DISABLED'})",
          file=sys.stderr)

# Canonical stack for cudaGraphLaunch host events. This is what HCTR's
# steady-state launch path always looks like (verified in source at
# HugeCTR/src/graph_wrapper.cpp:41 -- cudaGraphLaunch is called only
# from GraphWrapper::exec, which is invoked by GraphScheduleable::run,
# which is invoked by Pipeline::run_graph, which is invoked by
# Model::train_pipeline_with_ebc, which is invoked by Model::fit at
# pybind/model.cpp:884 -- the once-per-iter loop body).
#
# The LD_PRELOAD hook for cudaGraphLaunch in the shim could in principle
# verify this empirically, but it currently doesn't fire because libcudart
# resolves cudaGraphLaunch via ABI-versioning to _ptsz / _v10000 variants
# that the unversioned LD_PRELOAD export doesn't override. Source-level
# attribution is deterministic regardless.
GRAPH_LAUNCH_SRC_FUNCTION = "HugeCTR::GraphWrapper::exec"
GRAPH_LAUNCH_SRC_STACK_HCTR = [
    "HugeCTR::GraphWrapper::exec  (graph_wrapper.cpp:41)",
    "HugeCTR::GraphScheduleable::run",
    "HugeCTR::Pipeline::run_graph",
    "HugeCTR::Model::train_pipeline_with_ebc",
    "HugeCTR::Model::fit  (pybind/model.cpp:884)",
    "python3 train.py:488 (model.fit())",
]

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
# Find iteration boundaries.
#
# HCTR's per-iter schedule (HugeCTR/src/pybind/model_pipeline.cpp:387):
#   distribute_data -> ebc_mp_model_forward -> ebc_mp_network_forward
#   -> ebc_dp_forward -> network_graph (fwd+bwd+loss)
#   -> ebc_mp_backward_index_calculation -> ebc_dp_backward_index_calculation
#   -> distribute_data(next-iter prefetch)
#   -> ebc_mp_network_backward -> ebc_dp_local_reduce
#   -> network_exchange_wgrad (the main AllReduce)
#   -> ebc_dp_allreduce
#   -> update_params        (MLP optimizer = ada_grad_update4_kernel)
#   -> ebc_mp_local_reduce -> ebc_mp_update -> ebc_dp_update  (emb opt)
#   -> sync_back
#
# That means optimizers run AFTER AllReduce. Using AR end as the iter
# boundary therefore pushes each iter's opt to the start of the NEXT
# iter's window (misattribution). The compute-stream MLP optimizer
# (ada_grad_update4_kernel from HugeCTR/src/optimizers/adagrad_optimizer.cu)
# fires exactly ONCE per iter and is the last kernel on the compute
# stream within the iter, so its END is the correct iter boundary --
# optimizers now naturally land at the END of the iter window.
#
# Fallback: AllReduce_Sum_f16_RING_LL end (legacy behavior) if the
# adagrad kernel isn't found (e.g. a non-MLP-optimized variant).
# ------------------------------------------------------------------
REF_GPU = GPU_LIST[0]

# Primary marker: MLP optimizer kernel (1 per iter on compute stream)
opt_ids = [sid for sid, val in str_map.items()
           if val == "ada_grad_update4_kernel"]
iter_marker_source = None
ar_events = None
if opt_ids:
    ph = ",".join("?" * len(opt_ids))
    cur.execute(f"""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = ? AND shortName IN ({ph})
        ORDER BY start
    """, [REF_GPU] + opt_ids)
    opt_events = cur.fetchall()
    if opt_events:
        ar_events = opt_events
        iter_marker_source = "ada_grad_update4_kernel (MLP optimizer, end-of-iter)"
        print(f"GPU {REF_GPU} (ref): {len(opt_events)} ada_grad_update4_kernel "
              f"events -- using as end-of-iter marker", file=sys.stderr)

if ar_events is None:
    ar_ids = [sid for sid, val in str_map.items()
              if "AllReduce_Sum_f16_RING_LL" in val]
    ph = ",".join("?" * len(ar_ids))
    cur.execute(f"""
        SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL
        WHERE deviceId = ? AND shortName IN ({ph})
        ORDER BY start
    """, [REF_GPU] + ar_ids)
    ar_events = cur.fetchall()
    iter_marker_source = ("AllReduce_Sum_f16_RING_LL end "
                          "(FALLBACK -- optimizer will appear at start "
                          "of next iter)")
    print(f"GPU {REF_GPU} (ref): {len(ar_events)} AllReduce events "
          f"(fallback marker; ada_grad_update4_kernel not found)",
          file=sys.stderr)

if FIRST_ITER + N_ITERS >= len(ar_events):
    print(f"ERROR: only {len(ar_events)} iters", file=sys.stderr); sys.exit(2)

T_START = ar_events[FIRST_ITER - 1][1]
T_END   = ar_events[FIRST_ITER + N_ITERS - 1][1]
print(f"Window: iter {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}  "
      f"T_START={T_START}ns  T_END={T_END}ns  span={(T_END-T_START)/1e6:.2f}ms",
      file=sys.stderr)
print(f"  iter marker: {iter_marker_source}", file=sys.stderr)
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
            sfs = gemm_shape.get("src_functions", [])
            if sfs:
                args["src_function_candidates"] = sfs
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
                # Attach per-call src_function + HCTR-only stack frames
                # from the rich template (captured by the shim's
                # backtrace() at each cublasLtMatmul call during init).
                if pos < len(ITER_TEMPLATE_RICH):
                    rich = ITER_TEMPLATE_RICH[pos]
                    if rich.get("src_function"):
                        a["src_function"] = rich["src_function"]
                    if rich.get("stack_hctr"):
                        a["src_stack_hctr"] = rich["stack_hctr"]
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
                host_args = {"correlationId": corr, "globalTid": gtid,
                             "duration_us": round((end - start) / 1000.0, 2),
                             "gpu": GPU}
                # PyTorch-style stack propagation:
                # (a) If the kernel this host call launched has
                #     src_function/src_stack_hctr (set by the sequence
                #     matcher), copy them onto the host event so clicking
                #     a cublasLtMatmul / cudaLaunchKernel in Perfetto shows
                #     the same call-chain attribution as the kernel.
                # (b) For cudaGraphLaunch host events (the per-iter steady-
                #     state launcher in HCTR's captured-graph world), the
                #     per-call correlationId DOESN'T match any kernel's
                #     correlationId (graph-replayed kernels have IDs from
                #     graph-capture time). So attach the canonical
                #     hardcoded stack from GRAPH_LAUNCH_SRC_STACK_HCTR.
                kev_idx = kernel_by_corr.get(corr)
                if kev_idx is not None:
                    kev_args = events[kev_idx].get("args", {})
                    for k in ("src_function", "src_stack_hctr",
                              "dlrm_layer", "gemm_M", "gemm_N", "gemm_K",
                              "gemm_op_A", "gemm_op_B", "gemm_epilogue",
                              "gemm_gflops"):
                        if k in kev_args:
                            host_args[k] = kev_args[k]
                if "GraphLaunch" in n:
                    host_args["src_function"]   = GRAPH_LAUNCH_SRC_FUNCTION
                    host_args["src_stack_hctr"] = GRAPH_LAUNCH_SRC_STACK_HCTR
                ev = {
                    "name": n.replace("_v10000", "").replace("_v11010", "").replace("_v7000", ""),
                    "cat": "host_api",
                    "ph": "X",
                    "ts": (start - T_START) / 1000.0,
                    "dur": (end - start) / 1000.0,
                    "pid": PID_HOST, "tid": tid_lane,
                    "args": host_args,
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
            args2 = {"correlationId": corr, "globalTid": gtid,
                     "duration_us": round((end - start) / 1000.0, 2),
                     "gpu": GPU}
            # Same canonical-stack attachment as in the correlationId-join
            # path above -- so the cudaGraphLaunch host events that show
            # up here (the steady-state per-iter launchers) carry the
            # HCTR call chain in their Perfetto Args panel.
            if "GraphLaunch" in n:
                args2["src_function"]   = GRAPH_LAUNCH_SRC_FUNCTION
                args2["src_stack_hctr"] = GRAPH_LAUNCH_SRC_STACK_HCTR
            ev = {
                "name": n.replace("_v10000", "").replace("_v11010", "").replace("_v7000", ""),
                "cat": "host_api",
                "ph": "X",
                "ts": (start - T_START) / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": PID_HOST, "tid": lane_of(gtid),
                "args": args2,
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

    # ------------------------------------------------------------------
    # Tier A + C: SYNTHETIC NESTED CALL-STACK LANE
    # ------------------------------------------------------------------
    # PyTorch-profiler-style nested view on a single host lane (tid=500):
    #   iter N                    <-- outer slice spanning [iter_start, iter_end]
    #     mlp_fwd (23 kernels)    <-- phase slice spanning all phase kernels
    #       bot_mlp_L1_fwd        <-- layer slice (only for mlp/cross phases)
    #         cutlass3x... M=128 N=512 K=13 ...  <-- innermost kernel slice
    #
    # The hierarchy is reconstructed deterministically from:
    #   - iter boundaries (AR end timestamps)
    #   - per-kernel cat (already classified)
    #   - per-kernel dlrm_layer (set by GEMM catalog / sequence matcher)
    #   - per-kernel src_function + src_stack_hctr (from shim backtraces;
    #     attached to layer slices as args so clicking shows the call chain)
    #
    # Why this works even though steady-state never re-executes layer code:
    # graph capture happened ONCE during init and recorded exactly the
    # sequence of (layer-name -> kernel) the GEMM catalog stores. Graph
    # REPLAY just runs the same kernels in the same order; we re-attach
    # the layer/phase labels to those kernels via the sequence matcher.
    # ------------------------------------------------------------------
    # Tier-C synth lanes live in the host process. To keep nesting clean
    # (Chrome trace requires siblings on the same tid to be either fully
    # disjoint or fully nested -- partial overlap breaks Perfetto), each
    # GPU CUDA stream gets its OWN synth lane: the compute stream's lane
    # gets the full iter->phase->layer->kernel hierarchy, secondary streams
    # (embedding, a2a, sparse_prep, etc.) get a flatter iter->phase view
    # since they don't carry layer info. The compute stream is identified
    # as the one with the most kernels; its label was set by classify_stream.
    SYNTH_LANE_BASE = 500
    SYNTH_LANES = {}  # role -> tid
    def synth_lane_for(role):
        if role not in SYNTH_LANES:
            SYNTH_LANES[role] = SYNTH_LANE_BASE + len(SYNTH_LANES)
        return SYNTH_LANES[role]

    def _bucket_phase(cat, ts, loss_t):
        """Map a kernel category to a coarse phase label. Splits emb_a2a
        and interaction into fwd/bwd halves using the loss timestamp."""
        if cat == "sparse_prep":     return "sparse_prep"
        if cat in ("emb_fwd", "emb_reduce"):
            return "embedding_fwd"
        if cat == "emb_a2a":
            return ("embedding_a2a_bwd"
                    if loss_t is not None and ts > loss_t
                    else "embedding_a2a_fwd")
        if cat == "mlp_fwd":         return "mlp_fwd"
        if cat == "interaction":
            return ("interaction_bwd"
                    if loss_t is not None and ts > loss_t
                    else "interaction_fwd")
        if cat in ("mlp_bwd_dgrad", "mlp_bwd_wgrad"):
            return "mlp_bwd"
        if cat == "loss":            return "loss"
        if cat in ("emb_scatter", "emb_grad_reduce"):
            return "embedding_bwd"
        if cat == "allreduce":       return "allreduce"
        if cat in ("opt_emb", "opt_dense"):
            return "optimizer"
        if cat in ("memcpy", "memset"):
            return "memcpy"
        return "misc"

    PHASE_COLOR = {
        "sparse_prep":        "olive",
        "embedding_fwd":      "good",
        "embedding_a2a_fwd":  "rail_response",
        "embedding_a2a_bwd":  "rail_response",
        "embedding_bwd":      "good",
        "interaction_fwd":    "rail_idle_busy",
        "interaction_bwd":    "rail_idle_busy",
        "mlp_fwd":            "rail_animation",
        "mlp_bwd":            "rail_load",
        "loss":               "bad",
        "allreduce":          "bad",
        "optimizer":          "yellow",
        "misc":               "grey",
        "memcpy":             "grey",
    }

    def _clip_disjoint(intervals, outer_end_us):
        """Take list of (ts_us, te_us, payload) sorted by ts; return a list
        where each te_us is clipped to next entry's ts_us (or outer_end).
        Drops entries with non-positive duration. Required because Chrome
        trace nesting on a single tid demands fully disjoint or fully
        nested siblings -- partial overlap breaks Perfetto rendering."""
        out = []
        n = len(intervals)
        for i, (ts, te, payload) in enumerate(intervals):
            limit = intervals[i+1][0] if i+1 < n else outer_end_us
            te_clip = min(te, limit)
            if te_clip > ts:
                out.append((ts, te_clip, payload))
        return out

    # Partition kernels by their stream's semantic role. Each role gets its
    # own synth lane that is GUARANTEED serial in time (since a single CUDA
    # stream serializes its work). Cross-stream parallelism is preserved by
    # showing each stream on a separate synth lane -- no false serialization,
    # no broken nesting.
    #
    # We classify each stream inline here (using the already-populated
    # per_stream_cat_hist / per_stream_kernel_names) because the canonical
    # stream_label dict isn't built until later in the per-GPU loop.
    synth_sid_to_role = {}
    for _sid in sorted(per_stream_cat_hist.keys()):
        synth_sid_to_role[_sid] = classify_stream(
            per_stream_cat_hist.get(_sid, {}),
            per_stream_kernel_names.get(_sid, []),
            per_stream_memcpy_count.get(_sid, 0))
    sid_to_role = synth_sid_to_role
    role_kernel_idxs = defaultdict(list)
    for ix, e in enumerate(events):
        if e["ph"] != "X" or e.get("pid") != PID_GPU:
            continue
        if e.get("cat") in (None, "memcpy", "memset"):
            # memcpy events live on their own gpu stream lanes already
            continue
        sid = e["tid"]
        role = sid_to_role.get(sid, f"stream_{sid}")
        role_kernel_idxs[role].append(ix)

    # role priority order (only roles present get a lane)
    ROLE_PRIORITY = ["computation_stream", "wgrad_stream",
                     "embedding_mp", "embedding_dp", "sparse_prep"]

    synth_events = []
    n_iter_slices = 0
    n_phase_slices = 0
    n_layer_slices = 0
    n_kernel_inner_slices = 0

    # Bucket per-role kernels into per-iter sublists once
    role_iter_kernels = {}
    for role, all_idxs in role_kernel_idxs.items():
        bucketed = [[] for _ in range(N_ITERS)]
        for ix in all_idxs:
            kts = events[ix]["ts"] * 1000.0 + T_START  # back to ns
            for i in range(N_ITERS):
                if (ar_events[FIRST_ITER - 1 + i][1] <= kts
                        < ar_events[FIRST_ITER + i][1]):
                    bucketed[i].append(ix)
                    break
        role_iter_kernels[role] = bucketed

    # For each role with kernels: emit iter -> phase -> (layer -> kernel)
    # nested hierarchy on its own lane.
    for role in ROLE_PRIORITY + sorted(set(role_kernel_idxs) - set(ROLE_PRIORITY)):
        if role not in role_kernel_idxs:
            continue
        if not role_kernel_idxs[role]:
            continue
        lane_tid = synth_lane_for(role)
        for it_i in range(N_ITERS):
            idxs = role_iter_kernels[role][it_i]
            if not idxs:
                continue
            iter_label = FIRST_ITER + it_i
            iter_ts_us = (ar_events[FIRST_ITER - 1 + it_i][1] - T_START) / 1000.0
            iter_te_us = (ar_events[FIRST_ITER + it_i][1] - T_START) / 1000.0
            iter_dur_us = iter_te_us - iter_ts_us

            # loss_t comes from the GLOBAL loss kernel (only on compute stream)
            loss_t = None
            for ix in per_iter_kernels[it_i]:
                if events[ix]["cat"] == "loss":
                    loss_t = events[ix]["ts"]; break

            # Walk kernels in time order and emit phase slices as CONTIGUOUS
            # time-runs. The same logical phase can appear multiple times per
            # iter because the HCTR schedule interleaves (e.g. on the compute
            # stream: bot_mlp_fwd -> interaction (cross net) -> top_mlp_fwd ->
            # loss -> top_mlp_bwd -> interaction_bwd -> bot_mlp_bwd).
            # Grouping all "mlp_fwd" kernels into one phase span would make it
            # non-contiguous and collide with interaction_fwd; emitting
            # contiguous runs preserves time accuracy and keeps siblings
            # disjoint on the synth lane.
            idxs_sorted = sorted(idxs, key=lambda i: events[i]["ts"])
            phase_ivs = []  # list of (ts, te, (phase_key, [event_indices]))
            cur_phase = None
            cur_idxs = []
            cur_ts = cur_te = 0.0
            for ix in idxs_sorted:
                ph = _bucket_phase(events[ix]["cat"], events[ix]["ts"], loss_t)
                kts = events[ix]["ts"]
                kte = kts + events[ix]["dur"]
                if ph != cur_phase:
                    if cur_idxs:
                        phase_ivs.append((cur_ts, cur_te, (cur_phase, cur_idxs)))
                    cur_phase = ph
                    cur_idxs = [ix]
                    cur_ts = kts
                    cur_te = kte
                else:
                    cur_idxs.append(ix)
                    cur_te = max(cur_te, kte)
            if cur_idxs:
                phase_ivs.append((cur_ts, cur_te, (cur_phase, cur_idxs)))
            # Safety net: ensure siblings disjoint (single-stream serial
            # ordering already guarantees this except across stream-priority
            # induced overlaps, which can leak in if classify() is wrong).
            phase_ivs = _clip_disjoint(phase_ivs, iter_te_us)
            if not phase_ivs:
                continue

            # 1) per-role iter slice (spans only this role's kernels within
            # the iter -- gives a feel for "this stream's busy time this iter")
            role_iter_ts = phase_ivs[0][0]
            role_iter_te = max(p[1] for p in phase_ivs)
            role_iter_dur = role_iter_te - role_iter_ts
            synth_events.append({
                "name": f"iter {iter_label} [{role}]  "
                        f"({role_iter_dur:.1f} us, {len(idxs)} kernels)",
                "cat":  "synth_iter",
                "cname": "black",
                "ph":   "X",
                "ts":   role_iter_ts,
                "dur":  role_iter_dur,
                "pid":  PID_HOST,
                "tid":  lane_tid,
                "args": {"iter": iter_label, "role": role,
                         "n_kernels": len(idxs),
                         "duration_us": round(role_iter_dur, 3),
                         "gpu": GPU},
            })
            n_iter_slices += 1

            # 2) phase slices on this lane
            for ph_ts, ph_te, (ph_key, ph_idxs) in phase_ivs:
                ph_dur = ph_te - ph_ts
                synth_events.append({
                    "name": f"{ph_key}  ({len(ph_idxs)} kernels)",
                    "cat":  "synth_phase",
                    "cname": PHASE_COLOR.get(ph_key, "grey"),
                    "ph":   "X",
                    "ts":   ph_ts,
                    "dur":  ph_dur,
                    "pid":  PID_HOST,
                    "tid":  lane_tid,
                    "args": {"phase": ph_key, "iter": iter_label,
                             "role": role,
                             "n_kernels": len(ph_idxs),
                             "duration_us": round(ph_dur, 3),
                             "gpu": GPU},
                })
                n_phase_slices += 1

                # 3) layer slices (only for mlp/cross/interaction phases).
                # Same contiguous-run logic as phase grouping: a layer slice
                # ends as soon as the next kernel's dlrm_layer differs,
                # preventing non-contiguous layer spans.
                if ph_key not in ("mlp_fwd", "mlp_bwd",
                                  "interaction_fwd", "interaction_bwd"):
                    continue
                ph_idxs_sorted = sorted(ph_idxs, key=lambda i: events[i]["ts"])
                layer_ivs = []
                cur_layer = None
                cur_lidxs = []
                cur_lts = cur_lte = 0.0
                for ix in ph_idxs_sorted:
                    lname = events[ix]["args"].get(
                        "dlrm_layer", "") or "(unannotated)"
                    kts = events[ix]["ts"]
                    kte = kts + events[ix]["dur"]
                    if lname != cur_layer:
                        if cur_lidxs:
                            layer_ivs.append(
                                (cur_lts, cur_lte, (cur_layer, cur_lidxs)))
                        cur_layer = lname
                        cur_lidxs = [ix]
                        cur_lts = kts
                        cur_lte = kte
                    else:
                        cur_lidxs.append(ix)
                        cur_lte = max(cur_lte, kte)
                if cur_lidxs:
                    layer_ivs.append((cur_lts, cur_lte, (cur_layer, cur_lidxs)))
                layer_ivs = _clip_disjoint(layer_ivs, ph_te)

                for l_ts, l_te, (lname, lidxs) in layer_ivs:
                    l_dur = l_te - l_ts
                    rep = events[lidxs[0]]
                    largs = {"layer": lname, "iter": iter_label,
                             "phase": ph_key, "role": role,
                             "n_kernels": len(lidxs),
                             "duration_us": round(l_dur, 3),
                             "gpu": GPU}
                    if "src_function" in rep["args"]:
                        largs["src_function"] = rep["args"]["src_function"]
                    if "src_stack_hctr" in rep["args"]:
                        largs["src_stack_hctr"] = rep["args"]["src_stack_hctr"]
                    for k in ("gemm_M","gemm_N","gemm_K","gemm_op_A",
                              "gemm_op_B","gemm_epilogue","gemm_gflops"):
                        if k in rep["args"]:
                            largs[k] = rep["args"][k]
                    synth_events.append({
                        "name": f"{lname}  ({len(lidxs)}k, {l_dur:.1f} us)",
                        "cat":  "synth_layer",
                        "cname": PHASE_COLOR.get(ph_key, "grey"),
                        "ph":   "X",
                        "ts":   l_ts,
                        "dur":  l_dur,
                        "pid":  PID_HOST,
                        "tid":  lane_tid,
                        "args": largs,
                    })
                    n_layer_slices += 1

                    # 4) innermost per-kernel slices (only if disjoint)
                    kev_iv = sorted(((events[i]["ts"],
                                      events[i]["ts"] + events[i]["dur"], i)
                                     for i in lidxs), key=lambda x: x[0])
                    if any(kev_iv[i][1] > kev_iv[i+1][0]
                           for i in range(len(kev_iv) - 1)):
                        continue
                    clipped = _clip_disjoint(
                        [(t0, t1, i) for t0, t1, i in kev_iv], l_te)
                    for k_ts, k_te, ix in clipped:
                        kev = events[ix]
                        kargs = dict(kev.get("args", {}))
                        kn = kargs.get("kernel", "")
                        M = kargs.get("gemm_M")
                        if M is not None:
                            disp = (f"{kn[:32]}  M={M} N={kargs['gemm_N']} "
                                    f"K={kargs['gemm_K']} "
                                    f"epi={kargs.get('gemm_epilogue','')}")
                        else:
                            disp = kn[:64]
                        synth_events.append({
                            "name": disp,
                            "cat":  "synth_kernel",
                            "ph":   "X",
                            "ts":   k_ts,
                            "dur":  k_te - k_ts,
                            "pid":  PID_HOST,
                            "tid":  lane_tid,
                            "args": kargs,
                        })
                        n_kernel_inner_slices += 1

    events.extend(synth_events)
    print(f"  synth-stack: lanes={len(SYNTH_LANES)} "
          f"iter={n_iter_slices} phase={n_phase_slices} "
          f"layer={n_layer_slices} kernel={n_kernel_inner_slices}",
          file=sys.stderr)
    for role, tid in SYNTH_LANES.items():
        events.append({"name": "thread_name", "ph": "M",
                       "pid": PID_HOST, "tid": tid,
                       "args": {"name": f"synth_call_stack [{role}]"}})
        sort_key = (ROLE_PRIORITY.index(role) if role in ROLE_PRIORITY
                    else 50 + len(ROLE_PRIORITY))
        events.append({"name": "thread_sort_index", "ph": "M",
                       "pid": PID_HOST, "tid": tid,
                       "args": {"sort_index": sort_key}})

    # ------------------------------------------------------------------
    # Tier B: optional Python-tracer sidecar merge (when --py-trace=PATH
    # is supplied). The sidecar is produced by scripts/hctr_py_tracer.py
    # running inside train_mi350.py with HCTR_PY_TRACE=1. It contains a
    # standalone Chrome trace with Python frame slices in CLOCK_REALTIME
    # ns. We rewrite ts into the nsys-relative window scale using the
    # session's utcEpochNs anchor (loaded once outside the per-GPU loop)
    # and merge as events on a per-rank "python" lane.
    # ------------------------------------------------------------------
    # (merge happens once after the per-GPU loop -- see below)

    # Iter boundary instant markers (only emit once on the lowest GPU,
    # so they don't visually duplicate across rows)
    if not all_iter_marks_emitted:
        boundary_short = ("opt_end" if iter_marker_source
                          and "ada_grad" in iter_marker_source else "AR_end")
        for label, t in iter_marks:
            events.append({
                "name": f"=== iter {label} ({boundary_short}) ===",
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
    events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
                   "args": {"name": f"Host (rank {GPU})"}})
    # Interleave host+GPU per rank so each GPU's panes sit next to its host
    # launch process. Layout (low sort_index sorts first):
    #   rank 0: Host(rank 0) sort=0, GPU 0 sort=1
    #   rank 1: Host(rank 1) sort=2, GPU 1 sort=3
    #   ...
    events.append({"name": "process_sort_index", "ph": "M", "pid": PID_HOST, "tid": 0,
                   "args": {"sort_index": 2 * GPU}})
    events.append({"name": "process_sort_index", "ph": "M", "pid": PID_GPU, "tid": 0,
                   "args": {"sort_index": 2 * GPU + 1}})
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

# ----------------------------------------------------------------------
# Tier B merge: optional Python tracer sidecar.
#
# The sidecar must be a JSON file with two top-level keys:
#   {
#     "clock": "CLOCK_REALTIME",         # or "utc_epoch_ns"
#     "anchor_realtime_ns": <int>,       # ns since UTC epoch at tracer start
#     "traceEvents": [
#         {"name": ..., "ph": "X",
#          "ts_realtime_ns": <int>,      # CLOCK_REALTIME ns (UTC since epoch)
#          "dur_ns": <int>,
#          "rank": <int>,                # which DGX rank (0..7)
#          "args": {...}},
#         ...
#     ]
#   }
#
# We map each Python event's wall-clock ns to nsys-window-relative us by:
#     ts_window_us = (ts_realtime_ns - utcEpochNs_session
#                     + ANALYSIS_DETAILS.startTime
#                     - T_START) / 1000
# where utcEpochNs_session and startTime are pulled from the nsys sqlite.
# This assumes the Python tracer ran INSIDE the same nsys session.
# Python events appear on a per-rank "python" lane (tid=600+) in the
# corresponding host process.
# ----------------------------------------------------------------------
n_py_events = 0
if py_trace_path:
    try:
        cur.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
        utc_session_ns = cur.fetchone()[0]
        cur.execute("SELECT startTime FROM ANALYSIS_DETAILS LIMIT 1")
        nsys_start_ns = cur.fetchone()[0]
        nsys_origin_ns = utc_session_ns - nsys_start_ns  # ns where nsys ts=0
        with open(py_trace_path) as f:
            py_sidecar = json.load(f)
        py_evts = py_sidecar.get("traceEvents", [])
        print(f"\n=== Merging Python tracer sidecar: {py_trace_path} "
              f"({len(py_evts)} events) ===", file=sys.stderr)
        # Per-rank python lane numbering
        PY_LANE_BASE = 600
        py_lane_per_rank = {}
        for e in py_evts:
            ts_real = e.get("ts_realtime_ns")
            dur_ns  = e.get("dur_ns", 0)
            rank    = e.get("rank", 0)
            if ts_real is None:
                continue
            # rewrite into nsys window-relative us
            ts_window_us = (ts_real - nsys_origin_ns - T_START) / 1000.0
            dur_us = dur_ns / 1000.0
            # filter to events that fall inside the window
            if ts_window_us + dur_us < 0:
                continue
            if ts_window_us > (T_END - T_START) / 1000.0:
                continue
            PID_HOST = PID_HOST_BASE + rank
            tid = PY_LANE_BASE + py_lane_per_rank.setdefault(rank, 0)
            ev = {
                "name": e.get("name", "py_frame"),
                "cat":  "python",
                "cname": "rail_animation",
                "ph":   "X",
                "ts":   ts_window_us,
                "dur":  dur_us,
                "pid":  PID_HOST,
                "tid":  tid,
                "args": e.get("args", {}),
            }
            all_events.append(ev)
            n_py_events += 1
        # Name + sort python lanes
        for rank in py_lane_per_rank.keys():
            PID_HOST = PID_HOST_BASE + rank
            all_events.append({"name": "thread_name", "ph": "M",
                               "pid": PID_HOST, "tid": PY_LANE_BASE,
                               "args": {"name": f"python (rank {rank})"}})
            all_events.append({"name": "thread_sort_index", "ph": "M",
                               "pid": PID_HOST, "tid": PY_LANE_BASE,
                               "args": {"sort_index": 1}})
        print(f"  merged {n_py_events:,} Python frame events into trace",
              file=sys.stderr)
    except Exception as e:
        print(f"  WARNING: failed to merge py-trace '{py_trace_path}': {e}",
              file=sys.stderr)

print(f"\n=== Summary ===", file=sys.stderr)
print(f"  GPUs       : {len(GPU_LIST)} {GPU_LIST}", file=sys.stderr)
print(f"  Window     : iter {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}  "
      f"({(T_END-T_START)/1e6:.2f} ms)", file=sys.stderr)
print(f"  Kernels    : {total_kernels:,}", file=sys.stderr)
print(f"  Host APIs  : {total_host_apis:,}", file=sys.stderr)
print(f"  Launch arr : {total_launch_pairs:,}", file=sys.stderr)
print(f"  FwdBwd arr : {total_fwdbwd_pairs:,}", file=sys.stderr)
print(f"  Py frames  : {n_py_events:,}", file=sys.stderr)
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
