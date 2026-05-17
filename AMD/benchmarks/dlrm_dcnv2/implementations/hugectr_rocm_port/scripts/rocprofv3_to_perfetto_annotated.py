#!/usr/bin/env python3
"""
Convert N consecutive iterations of one GPU's events from a `rocprofv3` CSV
trace bundle to a Chrome / Perfetto JSON trace, with PyTorch-profiler-style
annotations.

This is the ROCm/HIP equivalent of NV's `nsys_to_perfetto_annotated.py`
(see ../../../NVIDIA/.../scripts/nsys_to_perfetto_annotated.py). It reads
the three CSV files emitted by:

    rocprofv3 --kernel-trace --hip-trace -d <out_dir> --output-format csv \
              --output-file <prefix> -- <command>

Produces a Perfetto-loadable JSON with:

  1. DLRM-DCNv2 semantic role on every kernel ([emb_a2a], [emb_fwd],
     [mlp_fwd], [mlp_bwd_dgrad], [mlp_bwd_wgrad], [allreduce], [opt_emb],
     [loss], [sparse_prep], [interaction], etc.) — visible as the kernel-name
     prefix and as Perfetto category colors.

  2. Per-kernel grid / workgroup / VGPR / SGPR / LDS in tooltip args (click any
     kernel). Native AMD equivalents of NV's regs / smem.

  3. Host-launch -> device-kernel arrows via rocprofv3 Correlation_Id (Chrome
     flow events; click a kernel and Perfetto highlights the originating
     hipLaunchKernel / hipGraphLaunch on the host row).

  4. Heuristic fwd/bwd pair arrows within each iter (uses BCE loss kernel as
     the fwd->bwd boundary, then LIFO-pairs mlp_fwd<->mlp_bwd_dgrad,
     interaction<->interaction, emb_fwd<->emb_scatter,
     emb_a2a fwd<->emb_a2a bwd).

  5. Iter boundary instant markers (detected via RCCL kernel timing gaps).

Iter detection on AMD: rocprofv3 reports all RCCL ops as
`ncclDevKernel_Generic_1` (no Tree/Ring/SendRecv distinction). We detect iter
boundaries by clustering RCCL launches: a gap > `--iter-gap-us`
microseconds (default 200 us) marks a new iter.

Usage:
    python3 rocprofv3_to_perfetto_annotated.py \
        <trace_prefix> <gpu_idx> [first_iter] [n_iters] [output.json]

Examples:
    # convert iters 5..7 of GPU 0
    python3 rocprofv3_to_perfetto_annotated.py \
        /home/chcai/trace_a2a_bs55296_5step/a2a_bs55296_5step 0 5 3
"""
import csv, json, os, re, sys, argparse
from collections import defaultdict

# Hot-path parser for hipBLASLt (Tensile) GEMM kernel names. Names look like:
#   Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x176x128_MI16x16x1_..._WG64_4_1
# We extract macro-tile (MT), MFMA shape (MI), dtype prefix, and A/B layout
# tokens so the converter can surface GEMM problem-size class in event args
# without requiring HCTR-side instrumentation or hipBLASLt log capture.
_RE_HIPBLASLT_MT = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")
_RE_HIPBLASLT_MI = re.compile(r"_MI(\d+)x(\d+)x(\d+)")
_RE_HIPBLASLT_LAYOUT = re.compile(r"^Cijk_(\w{4})_(\w{4})_")
_RE_HIPBLASLT_DTYPE = re.compile(r"^Cijk_\w{4}_\w{4}_([A-Z]{2,4})_([A-Z]{1,4})_")

def parse_hipblaslt_kernel_name(name):
    """Extract GEMM shape/dtype hints from a Tensile-generated kernel name.

    Returns dict with keys (any subset present):
      macro_tile_m, macro_tile_n, macro_tile_k     -- per-WG output tile
      mfma_m, mfma_n, mfma_k                         -- MFMA instruction shape
      layout_a, layout_b                             -- "Ailk"/"Alik" etc.
      dtype, dtype_compute                           -- "HHS_BH" -> ("HHS","BH")
    Returns {} if the name doesn't match the Tensile pattern.
    """
    if not name or not name.startswith("Cijk_"):
        return {}
    info = {}
    m = _RE_HIPBLASLT_MT.search(name)
    if m:
        info["macro_tile_m"] = int(m.group(1))
        info["macro_tile_n"] = int(m.group(2))
        info["macro_tile_k"] = int(m.group(3))
        info["macro_tile"] = f"{m.group(1)}x{m.group(2)}x{m.group(3)}"
    m = _RE_HIPBLASLT_MI.search(name)
    if m:
        info["mfma_m"] = int(m.group(1))
        info["mfma_n"] = int(m.group(2))
        info["mfma_k"] = int(m.group(3))
        info["mfma"] = f"{m.group(1)}x{m.group(2)}x{m.group(3)}"
    m = _RE_HIPBLASLT_LAYOUT.match(name)
    if m:
        info["layout_a"] = m.group(1)
        info["layout_b"] = m.group(2)
    m = _RE_HIPBLASLT_DTYPE.match(name)
    if m:
        info["dtype"] = m.group(1)
        info["dtype_compute"] = m.group(2)
    return info

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("trace_prefix",
                help="Path prefix of rocprofv3 outputs (without "
                     "_kernel_trace.csv suffix)")
ap.add_argument("gpu", type=int, help="GPU index (0..NGPU-1)")
ap.add_argument("first_iter", type=int, nargs="?", default=5,
                help="First iter to include (default 5, skips warm-up)")
ap.add_argument("n_iters", type=int, nargs="?", default=3,
                help="Number of iters to include (default 3)")
ap.add_argument("out", nargs="?", default=None, help="Output JSON path")
ap.add_argument("--iter-gap-us", type=float, default=2000.0,
                help="(Fallback) Time gap (us) between consecutive RCCL kernels "
                     "that marks an iter boundary, used only when "
                     "hipGraphLaunch is absent (default 2000 us)")
ap.add_argument("--iter-marker", default="hipGraphLaunch",
                choices=["hipGraphLaunch", "rccl"],
                help="API/kernel used as iter boundary marker. Use "
                     "'hipGraphLaunch' when CUDA graphs are enabled (one "
                     "launch per iter per GPU rank, very reliable). Use "
                     "'rccl' as fallback (less reliable on AMD because "
                     "rocprofv3 reports all RCCL ops with the same name).")
args = ap.parse_args()

KT_CSV  = args.trace_prefix + "_kernel_trace.csv"
HIP_CSV = args.trace_prefix + "_hip_api_trace.csv"
AGENT_CSV = args.trace_prefix + "_agent_info.csv"
# Optional: rocprofv3 --memory-copy-trace output (DMA H2D/D2H/D2D events
# that don't appear under --kernel-trace because they aren't KERNEL_DISPATCH
# events on ROCm). When present, we surface them as a "memcpy_h2d" /
# "memcpy_d2h" lane in the Perfetto JSON so the trace matches NV nsys's
# default memcpy lane visibility.
MEMCPY_CSV = args.trace_prefix + "_memory_copy_trace.csv"
# Optional: rocprofv3 --rccl-trace output (RCCL API calls: ncclAllReduce,
# ncclAllGather, ncclSend, ncclRecv, ncclGroupStart/End, ...).  schema has
# no nelements/datatype args (those would require HCTR-side roctx markers),
# but we use the Function counts + time ranges to (a) summarise per-window
# op-type breakdown as a marker event, and (b) tag the rccl kernel events
# by their stream role (rccl_dedicated -> allreduce; embedding_a2a ->
# alltoall; etc.) so the trace distinguishes operation types.
RCCL_CSV = args.trace_prefix + "_rccl_api_trace.csv"
# Optional: rocprofv3 --marker-trace output (ROCTX ranges/marks emitted
# by HCTR when HCTR_ROCTX=1, e.g. "iter_42", "fwd/bmlp", "bwd/tmlp",
# "comm/mlp_wgrad_allreduce"). Phase 20 (2026-05-16): used to show
# host-side pipeline-phase ranges nested under iter ranges, with the
# correlationId-derived launch flow arrows linking each phase's host
# ranges to its GPU kernels.
MARKER_CSV = args.trace_prefix + "_marker_api_trace.csv"
GPU = args.gpu
FIRST_ITER = args.first_iter
N_ITERS = args.n_iters
OUT = args.out or f"{args.trace_prefix}.gpu{GPU}.iter{FIRST_ITER}-{FIRST_ITER+N_ITERS}.json"

for p in (KT_CSV, HIP_CSV, AGENT_CSV):
    if not os.path.exists(p):
        sys.exit(f"ERROR: missing {p}")

# ----------------------------------------------------------------------------
# Map Agent_Id ("Agent N") -> GPU index using agent_info.csv
# ----------------------------------------------------------------------------
gpu_agents = []
with open(AGENT_CSV) as f:
    for row in csv.DictReader(f):
        if row.get("Agent_Type") == "GPU":
            gpu_agents.append(int(row["Logical_Node_Id"]))
gpu_agents.sort()
if GPU >= len(gpu_agents):
    sys.exit(f"ERROR: GPU {GPU} out of range (have {len(gpu_agents)} GPUs)")
TARGET_AGENT = f"Agent {gpu_agents[GPU]}"
print(f"GPU {GPU} -> {TARGET_AGENT} (of {len(gpu_agents)} agents)", file=sys.stderr)

# ----------------------------------------------------------------------------
# Classifier — DLRM-DCNv2 semantic roles for ROCm/HIP kernel names
# ----------------------------------------------------------------------------
def classify(name):
    """Return (category, short_role_label)."""
    if not name:
        return ("other", "?")
    n = name.lower()

    # ------- RCCL (rocprofv3 uses single name for all collective ops) -------
    if "ncclDevKernel" in name or "rccl" in n:
        # We can't distinguish AllReduce vs SendRecv from the name alone;
        # mark all as a generic "rccl" category so user can color-filter.
        return ("rccl", "ncclDev")

    # ------- HCTR sparse-prep / embedding kernels -------
    if any(t in n for t in ("swizzle_keys", "label_and_count", "compress_offset",
                             "split_feat_major", "get_keys_flag",
                             "get_unique_key_same_ev_size", "keys_to_indices",
                             "replicate_bucket_range", "count_keys_per_gpu",
                             "transpose_buckets", "compute_shard_ranges",
                             "concat_keys_and_bucket_range", "mp_cal_src_ptrs",
                             "bucket_range")):
        return ("sparse_prep", name.split("(")[0].split("::")[-1])
    if "radix_sort" in n or "rocprim" in n and "sort" in n:
        return ("sparse_prep", "rocprim::radix_sort")
    if "scan_kernel" in n or "device_scan" in n or ("rocprim" in n and "scan" in n):
        return ("sparse_prep", "rocprim::scan")
    if "rocprim" in n:  # other rocprim primitives (reduce, transform, ...)
        return ("sparse_prep", "rocprim::primitive")
    if "ragged_static_embedding_table_lookup" in n:
        return ("emb_fwd", "ragged_static_embedding_lookup")
    if "multi_to_one_reduce" in n:
        return ("emb_reduce", "embedding::multi_to_one_reduce")
    if "multi_to_one_warp_per_ev" in n or "multi_to_one_warp" in n:
        return ("emb_fwd", "embedding::multi_to_one_warp")
    if "one_to_multi_warp_per_ev" in n or "one_to_multi" in n:
        return ("emb_scatter", "embedding::one_to_multi_warp")
    if "update4_kernel" in n:
        return ("opt_emb", "embedding::adagrad_update4")
    if "ada_grad_update" in n or "adagrad" in n:
        return ("opt_dense", "ada_grad_update")

    # ------- HCTR fused MLP custom kernels -------
    if "bgrada_v5" in n:
        return ("mlp_bwd_dgrad", "HugeCTR::bgrada_v5")
    if "drelu" in n or "_bgrad_" in n or "bgrad" in n:
        return ("mlp_bwd_dgrad", name.split("(")[0])
    if "add_bias_per_row" in n or "add_bias" in n:
        return ("mlp_fwd", "HugeCTR::add_bias_per_row")
    if "bias_relu" in n or "bias_f16_relu" in n:
        return ("mlp_fwd", name.split("(")[0])
    if "concat_fwd" in n:
        return ("interaction", "HugeCTR::concat_fwd")
    if "concat_bwd" in n:
        return ("interaction", "HugeCTR::concat_bwd")
    if "convert_array" in n or "transform_array" in n:
        return ("dtype_cast", "HugeCTR::convert_array")
    if "vector_mul_fma" in n or "vector_fma" in n or "fma3_align" in n:
        return ("fused_fma", "HugeCTR::vector_fma")

    # ------- BCE loss / regularizer -------
    if "binary_cross_entropy" in n or "binarycrossentropy" in n or "bce" in n:
        return ("loss", "BCE")

    # ------- hipBLASLt FP16 fused GEMM (Cijk_*) -------
    # Naming convention (Tensile-generated):
    #   Cijk_<A_layout>_<B_layout>_<types>_<...>_MT<M>x<N>x<K>_MI<...>
    # A=Ailk, B=Bjlk: A is row-major M-major, B is row-major K-major (FWD pattern)
    # A=Alik, B=Bljk or A=Ailk, B=Bljk: backward pattern (transposed inputs)
    if name.startswith("Cijk_"):
        if "_Ailk_Bjlk_" in name:
            return ("mlp_fwd", "hipBLASLt::gemm_fwd")
        if "_Alik_Bljk_" in name:
            return ("mlp_bwd_wgrad", "hipBLASLt::gemm_wgrad")
        if "_Ailk_Bljk_" in name:
            return ("mlp_bwd_dgrad", "hipBLASLt::gemm_dgrad")
        return ("mlp_fwd", "hipBLASLt::gemm")  # fallback
    if "splitkreduce" in n.replace("_", "") or "splitk_reduce" in n:
        return ("mlp_fwd", "hipBLASLt::splitK_reduce")

    # ------- Memory ops -------
    if "copybuffer" in n.replace("_", ""):
        return ("memcpy", "amd_rocclr_copyBuffer")
    if "fillbuffer" in n.replace("_", "") or "fillaligned" in n.replace("_", ""):
        return ("memset", "amd_rocclr_fillBuffer")
    if "memcpy" in n: return ("memcpy", "memcpy")
    if "memset" in n: return ("memset", "memset")

    # ------- HCTR generic fallback -------
    if "HugeCTR::" in name or "HugeCTR" in name:
        return ("hctr_other", name.split("(")[0].split("::")[-1])
    if "embedding::" in name:
        return ("emb_other", name.split("(")[0].split("::")[-1])

    return ("other", name.split("(")[0])

# Color palette (Chrome trace cnames; same scheme as NV's script)
COLOR = {
    "sparse_prep":   "olive",
    "emb_fwd":       "good",
    "emb_reduce":    "good",
    "emb_a2a":       "rail_response",
    "emb_scatter":   "good",
    "rccl":          "rail_response",
    "mlp_fwd":       "rail_animation",
    "interaction":   "rail_idle_busy",
    "mlp_bwd_dgrad": "rail_load",
    "mlp_bwd_wgrad": "rail_load",
    "allreduce":     "bad",
    "loss":          "bad",
    "opt_emb":       "yellow",
    "opt_dense":     "yellow",
    "dtype_cast":    "grey",
    "fused_fma":     "grey",
    "memcpy":        "grey",
    "memcpy_h2d":    "thread_state_iowait",
    "memcpy_d2h":    "thread_state_runnable",
    "memcpy_d2d":    "grey",
    "rccl_allreduce":  "rail_response",
    "rccl_allgather":  "rail_response",
    "rccl_alltoall":   "rail_response",
    "rccl_collective": "rail_response",
    "memset":        "grey",
    "hctr_other":    "white",
    "emb_other":     "white",
    "other":         "white",
}

PID_GPU  = 1
PID_HOST = 0

# ----------------------------------------------------------------------------
# 1) Load all kernel events for the target GPU
# ----------------------------------------------------------------------------
print(f"Loading {KT_CSV} ...", file=sys.stderr)
all_kernels = []  # list of dicts (sorted by start_ns)
rccl_starts = []  # for iter detection
with open(KT_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["Agent_Id"] != TARGET_AGENT:
            continue
        try:
            start = int(row["Start_Timestamp"])
            end   = int(row["End_Timestamp"])
        except (ValueError, KeyError):
            continue
        name = row["Kernel_Name"]
        sid  = int(row.get("Stream_Id", 0) or 0)
        tid  = int(row.get("Thread_Id", 0) or 0)
        corr = int(row.get("Correlation_Id", 0) or 0)
        wgx  = int(row.get("Workgroup_Size_X", 0) or 0)
        wgy  = int(row.get("Workgroup_Size_Y", 0) or 0)
        wgz  = int(row.get("Workgroup_Size_Z", 0) or 0)
        gx   = int(row.get("Grid_Size_X", 0) or 0)
        gy   = int(row.get("Grid_Size_Y", 0) or 0)
        gz   = int(row.get("Grid_Size_Z", 0) or 0)
        vgpr = int(row.get("VGPR_Count", 0) or 0)
        sgpr = int(row.get("SGPR_Count", 0) or 0)
        lds  = int(row.get("LDS_Block_Size", 0) or 0)  # bytes
        scratch = int(row.get("Scratch_Size", 0) or 0)
        threads_per_block = wgx * wgy * wgz
        # rocprofv3 grid sizes are in THREADS, not workgroups.
        blocks_x = gx // wgx if wgx else gx
        blocks_y = gy // wgy if wgy else gy
        blocks_z = gz // wgz if wgz else gz
        blocks_total = blocks_x * blocks_y * blocks_z
        cat, role = classify(name)
        all_kernels.append({
            "start": start, "end": end, "sid": sid, "tid": tid,
            "name": name, "cat": cat, "role": role, "corr": corr,
            "wgx": wgx, "wgy": wgy, "wgz": wgz,
            "gx_blocks": blocks_x, "gy_blocks": blocks_y, "gz_blocks": blocks_z,
            "blocks_total": blocks_total, "threads_per_block": threads_per_block,
            "vgpr": vgpr, "sgpr": sgpr, "lds": lds, "scratch": scratch,
        })
        if cat == "rccl":
            rccl_starts.append(start)

all_kernels.sort(key=lambda k: k["start"])
rccl_starts.sort()
print(f"  total kernels on {TARGET_AGENT}: {len(all_kernels):,}", file=sys.stderr)
print(f"  RCCL kernels: {len(rccl_starts):,}", file=sys.stderr)

# ----------------------------------------------------------------------------
# 2) Detect iter boundaries
#
# Preferred: hipGraphLaunch — when HCTR_USE_CUDA_GRAPH=1 (default), each iter
# is wrapped in a single graph launch per GPU rank, giving us EXACTLY one
# hipGraphLaunch event per iter per GPU. We identify "the host TID that owns
# GPU N" by finding which TIDs issued kernels for our target Agent.
#
# Fallback: RCCL kernel time gaps (less reliable — rocprofv3 reports all RCCL
# collectives as `ncclDevKernel_Generic_1` and there are 5–7 RCCL kernels per
# iter on AMD).
# ----------------------------------------------------------------------------
iter_ends = []  # ns timestamps, one per iter end

if args.iter_marker == "hipGraphLaunch":
    # Find host TIDs that issued kernels for our target GPU.
    target_tids = {k["tid"] for k in all_kernels if k["tid"] > 0}
    print(f"  candidate host TIDs for {TARGET_AGENT}: {sorted(target_tids)}",
          file=sys.stderr)
    # Read hipGraphLaunch events from HIP API trace and pick those issued by
    # a TID in target_tids.
    graph_starts_per_tid = defaultdict(list)
    with open(HIP_CSV) as f:
        for row in csv.DictReader(f):
            if row.get("Function") != "hipGraphLaunch":
                continue
            tid = int(row["Thread_Id"])
            if tid not in target_tids:
                continue
            graph_starts_per_tid[tid].append(int(row["End_Timestamp"]))
    # Pick the TID with the most graph-launches (the one actually driving GPU 0)
    if not graph_starts_per_tid:
        print("  WARNING: no hipGraphLaunch events on target TIDs; "
              "falling back to RCCL gaps", file=sys.stderr)
        args.iter_marker = "rccl"
    else:
        owner_tid = max(graph_starts_per_tid, key=lambda t: len(graph_starts_per_tid[t]))
        graph_starts = sorted(graph_starts_per_tid[owner_tid])
        # iter N "ends" at the START of graph_launch N+1 (or the graph_end for
        # the last one). Simpler: use start of each graph_launch as iter
        # boundary (the iter that just finished ends here).
        iter_ends = graph_starts
        print(f"  iter detection via hipGraphLaunch on TID {owner_tid}: "
              f"{len(iter_ends)} iters", file=sys.stderr)

if not iter_ends:  # fallback or explicit "rccl" mode
    GAP_NS = int(args.iter_gap_us * 1000)
    if rccl_starts:
        cur_end = rccl_starts[0]
        for t in rccl_starts[1:]:
            if t - cur_end > GAP_NS:
                iter_ends.append(cur_end)
            cur_end = t
        iter_ends.append(cur_end)
    print(f"  iter detection via RCCL gap > {args.iter_gap_us:.0f} us: "
          f"{len(iter_ends)} iters", file=sys.stderr)

if FIRST_ITER + N_ITERS > len(iter_ends):
    sys.exit(f"ERROR: requested iters {FIRST_ITER}..{FIRST_ITER+N_ITERS} but "
             f"only detected {len(iter_ends)} iters")

T_START = iter_ends[FIRST_ITER - 1] if FIRST_ITER > 0 else iter_ends[0]
T_END   = iter_ends[FIRST_ITER + N_ITERS - 1]
print(f"  window: iter {FIRST_ITER}..{FIRST_ITER+N_ITERS-1}  "
      f"T_START={T_START}ns span={(T_END-T_START)/1e6:.3f}ms", file=sys.stderr)

iter_marks = [(FIRST_ITER + i, iter_ends[FIRST_ITER - 1 + i] if FIRST_ITER > 0
               else iter_ends[i])
              for i in range(N_ITERS + 1)]

# ----------------------------------------------------------------------------
# 3) Build kernel events for the window + per-iter buckets
# ----------------------------------------------------------------------------
events = []
kernel_by_corr = {}    # Correlation_Id -> ev_idx
per_iter_kernels = [[] for _ in range(N_ITERS)]
cat_count = defaultdict(int)

for k in all_kernels:
    if k["start"] < T_START or k["start"] >= T_END:
        continue
    # which iter? iter `FIRST_ITER + i` runs from iter_ends[FIRST_ITER-1+i]
    # to iter_ends[FIRST_ITER+i].
    iter_idx = -1
    for i in range(N_ITERS):
        lo = iter_ends[FIRST_ITER - 1 + i]
        hi = iter_ends[FIRST_ITER + i]
        if lo <= k["start"] < hi:
            iter_idx = i
            break

    cat_count[k["cat"]] += 1
    short_kn = k["name"][:60] if len(k["name"]) > 60 else k["name"]
    # Parse hipBLASLt GEMM kernel name for macro-tile/MFMA/dtype hints. For
    # MLP forward/backward GEMMs these go straight into the display string
    # so each event's lane label shows e.g. [mlp_fwd] MT128x176x128 (HHS).
    gemm_info = parse_hipblaslt_kernel_name(k["name"])
    if gemm_info.get("macro_tile"):
        gemm_label = f"MT{gemm_info['macro_tile']}"
        if gemm_info.get("dtype"):
            gemm_label += f" ({gemm_info['dtype']})"
        display = f"[{k['cat']}] hipBLASLt {gemm_label}"
    else:
        display = (f"[{k['cat']}] {k['role']}" if k["role"] and k["role"] != k["name"]
                   else f"[{k['cat']}] {short_kn}")

    ev_args = {
        "kernel": k["name"],
        "iter": (FIRST_ITER + iter_idx) if iter_idx >= 0 else "",
        "duration_us": round((k["end"] - k["start"]) / 1000.0, 3),
        # PyTorch-profiler-style launch args (AMD-native)
        "grid_workgroups": f"{k['gx_blocks']}x{k['gy_blocks']}x{k['gz_blocks']}",
        "workgroup_size":  f"{k['wgx']}x{k['wgy']}x{k['wgz']}",
        "blocks_total":    k["blocks_total"],
        "threads_per_block": k["threads_per_block"],
        "vgpr_per_thread":  k["vgpr"],
        "sgpr_count":       k["sgpr"],
        "lds_bytes":        k["lds"],
        "scratch_bytes":    k["scratch"],
        "correlationId":    k["corr"],
    }
    if gemm_info:
        # Surface the parsed GEMM shape directly in the event args so users
        # can sort/filter by macro-tile or MFMA shape in Perfetto.
        ev_args.update(gemm_info)
        # Best-effort problem-size class: workgroups * tile_m * tile_n is
        # a proxy for output flops (modulo K, batch, and split-K). Real
        # M/N/K would need hipBLASLt log capture (HIPBLASLT_LOG_FILE=...).
        wg_total = k["blocks_total"]
        if (gemm_info.get("macro_tile_m") and gemm_info.get("macro_tile_n")):
            ev_args["estimated_output_elems"] = (
                wg_total * gemm_info["macro_tile_m"]
                         * gemm_info["macro_tile_n"]
            )

    ev = {
        "name": display,
        "cat": k["cat"],
        "ph": "X",
        "ts": (k["start"] - T_START) / 1000.0,   # us
        "dur": (k["end"] - k["start"]) / 1000.0, # us
        "pid": PID_GPU,
        "tid": k["sid"],
        "args": ev_args,
    }
    if k["cat"] in COLOR:
        ev["cname"] = COLOR[k["cat"]]
    ev_idx = len(events)
    events.append(ev)
    if k["corr"]:
        kernel_by_corr[k["corr"]] = ev_idx
    if iter_idx >= 0:
        per_iter_kernels[iter_idx].append(ev_idx)

n_kern_in_window = sum(len(x) for x in per_iter_kernels)
print(f"  kernels in window: {n_kern_in_window:,}", file=sys.stderr)
print(f"  category breakdown: {dict(cat_count)}", file=sys.stderr)

# ----------------------------------------------------------------------------
# 3b) Load MEMORY_COPY events from rocprofv3 --memory-copy-trace output
#
# These are HSA-level DMA transfers (H2D / D2H / D2D). They are NOT captured
# by --kernel-trace (which only sees KERNEL_DISPATCH ops), so without this
# step the trace appears to lack a memcpy lane even though the data reader's
# placement_streams_ are firing per-iter (matches NV nsys's default memcpy
# lane). Filtered to the same iter window + target Agent as kernels.
# Schema: Kind,Direction,Stream_Id,Source_Agent_Id,Destination_Agent_Id,
#         Correlation_Id,Start_Timestamp,End_Timestamp
# ----------------------------------------------------------------------------
n_memcpy_in_window = 0
if os.path.exists(MEMCPY_CSV):
    print(f"Loading {MEMCPY_CSV} (DMA memcpy events) ...", file=sys.stderr)
    memcpy_count_by_dir_stream = defaultdict(int)
    with open(MEMCPY_CSV) as f:
        for row in csv.DictReader(f):
            direction = row.get("Direction", "")
            src = row.get("Source_Agent_Id", "")
            dst = row.get("Destination_Agent_Id", "")
            # filter: keep events whose source OR destination is our target GPU
            if src != TARGET_AGENT and dst != TARGET_AGENT:
                continue
            try:
                start = int(row["Start_Timestamp"])
                end   = int(row["End_Timestamp"])
            except (ValueError, KeyError):
                continue
            if start < T_START or start >= T_END:
                continue
            sid = int(row.get("Stream_Id", 0) or 0)
            corr = int(row.get("Correlation_Id", 0) or 0)
            # short label: H2D / D2H / D2D
            if "HOST_TO_DEVICE" in direction:
                short_dir = "H2D"
                cat = "memcpy_h2d"
            elif "DEVICE_TO_HOST" in direction:
                short_dir = "D2H"
                cat = "memcpy_d2h"
            elif "DEVICE_TO_DEVICE" in direction:
                short_dir = "D2D"
                cat = "memcpy_d2d"
            else:
                short_dir = direction
                cat = "memcpy"
            # which iter
            iter_idx = -1
            for i in range(N_ITERS):
                lo = iter_ends[FIRST_ITER - 1 + i]
                hi = iter_ends[FIRST_ITER + i]
                if lo <= start < hi:
                    iter_idx = i
                    break
            src_clean = src.replace('"', '')
            dst_clean = dst.replace('"', '')
            events.append({
                "name": f"[{cat}] {short_dir} {src_clean}->{dst_clean}",
                "cat": cat,
                "ph": "X",
                "ts": (start - T_START) / 1000.0,
                "dur": (end - start) / 1000.0,
                "pid": PID_GPU,
                "tid": sid,
                "args": {
                    "direction": direction,
                    "source": src,
                    "destination": dst,
                    "duration_us": round((end - start) / 1000.0, 3),
                    "correlationId": corr,
                    "iter": (FIRST_ITER + iter_idx) if iter_idx >= 0 else "",
                },
                "cname": "grey",
            })
            n_memcpy_in_window += 1
            memcpy_count_by_dir_stream[(short_dir, sid)] += 1
    print(f"  DMA memcpy events in window (on {TARGET_AGENT}): {n_memcpy_in_window:,}",
          file=sys.stderr)
    if n_memcpy_in_window:
        print(f"  --- memcpy count by (direction, stream) ---", file=sys.stderr)
        for (d, sid), c in sorted(memcpy_count_by_dir_stream.items(),
                                  key=lambda x: -x[1])[:10]:
            print(f"    {d}  stream {sid:5d}  count={c}", file=sys.stderr)
else:
    print(f"  (no {MEMCPY_CSV} present -- run rocprofv3 with "
          f"--memory-copy-trace to populate the memcpy lane)", file=sys.stderr)

# ----------------------------------------------------------------------------
# 4) Host-side HIP API events + launch flow arrows
# ----------------------------------------------------------------------------
print(f"Loading {HIP_CSV} (host API events) ...", file=sys.stderr)

# Determine the "owner" host thread for this GPU: the host TID that issued
# the most kernels for our target GPU agent. NV trace convention: only one
# host thread per GPU section. We filter out all other host TIDs to keep
# the trace lane count low and the visualization clean.
host_tid_kernel_count = defaultdict(int)
for k in all_kernels:
    if k["tid"] > 0:
        host_tid_kernel_count[k["tid"]] += 1
gpu_owner_host_tid = (max(host_tid_kernel_count, key=host_tid_kernel_count.get)
                     if host_tid_kernel_count else 0)
print(f"  GPU owner host TID: {gpu_owner_host_tid} "
      f"({host_tid_kernel_count.get(gpu_owner_host_tid, 0)} kernels)",
      file=sys.stderr)

# Map host TIDs to small lane indices (per process). Only the owner gets a lane.
host_tid_to_lane = {}
def lane_of(tid):
    if tid not in host_tid_to_lane:
        host_tid_to_lane[tid] = len(host_tid_to_lane) + 1
    return host_tid_to_lane[tid]

# Heuristic: API events are interesting if their corr matches a kernel in our window
# OR they're in the time window AND on a thread that owns a kernel in the window.
window_corrs = set(kernel_by_corr.keys())
launch_pairs = 0
host_api_added = 0
host_api_skipped_nonowner = 0

with open(HIP_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            corr = int(row.get("Correlation_Id", 0) or 0)
            start = int(row["Start_Timestamp"])
            end   = int(row["End_Timestamp"])
            tid   = int(row.get("Thread_Id", 0) or 0)
        except (ValueError, KeyError):
            continue
        fname  = row.get("Function", "?")
        domain = row.get("Domain", "")
        # Skip noise: __hipRegister*, Init, ModuleLoad, etc.
        if fname.startswith("__hipRegister") or fname in (
                "hipInit", "hipDeviceGetCount", "hipDeviceGet",
                "hipGetDeviceProperties", "hipDriverGetVersion",
                "hipRuntimeGetVersion"):
            continue
        # Only keep API events that:
        #   (a) match a kernel correlationId in our window (host launch), or
        #   (b) fall in the time window AND are interesting sync/launch APIs
        match = corr in window_corrs
        in_window = (T_START <= start < T_END)
        is_interesting = any(t in fname for t in (
            "LaunchKernel", "GraphLaunch", "MemcpyAsync", "Memcpy",
            "StreamSync", "EventSync", "DeviceSync"))
        if not match and not (in_window and is_interesting):
            continue
        # Only keep host events from THIS GPU's owner host thread. Other host
        # threads' events would clutter the host lane with unrelated activity
        # from sibling GPUs (each GPU has its own owner). Match NV's nsys
        # convention of one host_thread per GPU section.
        if tid != gpu_owner_host_tid:
            host_api_skipped_nonowner += 1
            continue

        cname = ("bad" if "GraphLaunch" in fname
                 else "rail_idle_busy" if "Sync" in fname
                 else None)
        ev = {
            "name": fname,
            "cat": "host_api",
            "ph": "X",
            "ts": (start - T_START) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_HOST, "tid": lane_of(tid),
            "args": {"correlationId": corr, "thread_id": tid,
                     "duration_us": round((end - start) / 1000.0, 3)},
        }
        if cname:
            ev["cname"] = cname
        events.append(ev)
        host_api_added += 1

        # Emit host -> device launch flow arrow if this API caused a kernel
        if match:
            kev = events[kernel_by_corr[corr]]
            host_t1 = (end - T_START) / 1000.0
            events.append({
                "name": "launch", "cat": "launch",
                "ph": "s", "id": corr,
                "ts": host_t1,
                "pid": PID_HOST, "tid": lane_of(tid),
            })
            events.append({
                "name": "launch", "cat": "launch",
                "ph": "f", "id": corr,
                "ts": kev["ts"],
                "pid": kev["pid"], "tid": kev["tid"],
                "bp": "e",
            })
            launch_pairs += 1

print(f"  host API events kept: {host_api_added:,}", file=sys.stderr)
print(f"  host API events skipped (non-owner TID): {host_api_skipped_nonowner:,}",
      file=sys.stderr)
print(f"  host->device launch arrows: {launch_pairs:,}", file=sys.stderr)
print(f"  host threads (lanes): {len(host_tid_to_lane)}", file=sys.stderr)

# ----------------------------------------------------------------------------
# 4b) ROCTX marker ranges (Phase 20, HCTR_ROCTX=1)
# ----------------------------------------------------------------------------
# rocprofv3 --marker-trace produces marker_api_trace.csv with the schema:
#   Domain,Function,Process_Id,Thread_Id,Correlation_Id,
#   Start_Timestamp,End_Timestamp
# For our HCTR ranges, Domain="MARKER_CORE_API" and Function is the
# message we pushed (e.g. "iter_42", "fwd/bmlp", "comm/mlp_wgrad_allreduce").
# We render each range as a Perfetto X event on its own lane on the
# owner host process, with a category prefix-derived color so iter /
# fwd / bwd / comm / opt stripes are visually distinct.
marker_added = 0
marker_skipped_outside = 0
if os.path.exists(MARKER_CSV):
    print(f"Loading {MARKER_CSV} (ROCTX marker ranges) ...", file=sys.stderr)
    # Pick a lane index dedicated to roctx ranges so they don't intermix
    # with hipLaunchKernel events on the host_thread lane.
    roctx_lane = lane_of(-1)  # synthetic tid -1 -> next free lane
    def _color_for(prefix):
        # Perfetto reserved color names ("good", "bad", "thread_state_*",
        # etc.) -- pick distinct colors per phase category.
        return {
            "iter":  "thread_state_runnable",
            "fwd":   "rail_response",     # green-ish
            "bwd":   "rail_animation",    # orange-ish
            "comm":  "bad",               # red-ish
            "opt":   "thread_state_running",
            "data":  "rail_idle_busy",    # blue-ish
            "sync_back": "grey",
            "graph": "yellow",
        }.get(prefix, None)
    with open(MARKER_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                start = int(row["Start_Timestamp"])
                end   = int(row["End_Timestamp"])
                tid   = int(row.get("Thread_Id", 0) or 0)
            except (ValueError, KeyError):
                continue
            domain = row.get("Domain", "")
            name   = row.get("Function", "?")
            # Filter to HCTR-emitted ranges and the active iter window.
            # rocprofv3 emits ROCTX push/pop ranges in domain
            # MARKER_CORE_RANGE_API, instant marks in MARKER_CORE_API,
            # and the SDK control calls (roctxProfilerPause etc.) in
            # MARKER_CONTROL_API. We want the first two but not API
            # internals.
            if name.startswith("roctx"):
                continue
            if domain not in ("MARKER_CORE_RANGE_API", "MARKER_CORE_API"):
                continue
            if end <= T_START or start >= T_END:
                marker_skipped_outside += 1
                continue
            prefix = name.split("/", 1)[0] if "/" in name else (
                "iter" if name.startswith("iter_") else "")
            cname = _color_for(prefix) or _color_for("graph")
            ev = {
                "name": name,
                "cat":  "roctx",
                "ph":   "X",
                "ts":   (max(start, T_START) - T_START) / 1000.0,
                "dur":  (min(end, T_END) - max(start, T_START)) / 1000.0,
                "pid":  PID_HOST, "tid": roctx_lane,
                "args": {"phase_prefix": prefix, "thread_id": tid,
                         "duration_us": round((end - start) / 1000.0, 3)},
            }
            if cname:
                ev["cname"] = cname
            events.append(ev)
            marker_added += 1
    # Label the lane
    events.append({"name": "thread_name", "ph": "M",
                   "pid": PID_HOST, "tid": roctx_lane,
                   "args": {"name": "roctx_ranges (HCTR phases)"}})
    print(f"  ROCTX marker ranges kept: {marker_added:,} "
          f"(skipped outside window: {marker_skipped_outside:,})",
          file=sys.stderr)
else:
    print(f"  (no MARKER_CSV at {MARKER_CSV}; skipping ROCTX overlay)",
          file=sys.stderr)

# ----------------------------------------------------------------------------
# 5) Heuristic forward/backward kernel pairing arrows
# ----------------------------------------------------------------------------
# Per iter:
#   - find the BCE loss kernel (one per iter); kernels before it = fwd,
#     after it = bwd
#   - bucket by (cat, phase), LIFO-pair by category
#
# For RCCL (emb_a2a + allreduce all share "rccl" category on AMD):
#   - first half of RCCL kernels = a2a fwd, second half = a2a bwd
#     (heuristic; final RCCL may also be the AllReduce)
PAIRS = [
    ("mlp_fwd",     "mlp_bwd_dgrad"),
    ("mlp_fwd",     "mlp_bwd_wgrad"),
    ("interaction", "interaction"),
    ("emb_fwd",     "emb_scatter"),
    ("rccl",        "rccl"),
]

def kernel_phase(ev, loss_t):
    return "fwd" if ev["ts"] + ev["dur"] <= loss_t else "bwd"

flow_pair_count = 0
flow_id = 1_000_000_000

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
        # fallback: midpoint
        first_t = events[idxs[0]]["ts"]
        last_t  = events[idxs[-1]]["ts"] + events[idxs[-1]]["dur"]
        loss_t = (first_t + last_t) / 2

    bucket = defaultdict(list)
    for ix in idxs:
        e = events[ix]
        bucket[(e["cat"], kernel_phase(e, loss_t))].append(ix)

    for cat_fwd, cat_bwd in PAIRS:
        fwds = bucket.get((cat_fwd, "fwd"), [])
        bwds = bucket.get((cat_bwd, "bwd"), [])
        if cat_fwd == cat_bwd:
            # within-category pairing: split list at midpoint instead
            all_in_cat = fwds + bwds
            half = len(all_in_cat) // 2
            fwds = all_in_cat[:half]
            bwds = all_in_cat[half:]
        for i in range(min(len(fwds), len(bwds))):
            fix = fwds[-(i + 1)]
            bix = bwds[i]
            fe, be = events[fix], events[bix]
            flow_id += 1
            cat_label = f"fwd_bwd_{cat_fwd}"
            events.append({
                "name": f"{cat_fwd} -> {cat_bwd} pair",
                "cat":  cat_label,
                "ph":   "s", "id": flow_id,
                "ts":   fe["ts"] + fe["dur"],
                "pid":  fe["pid"], "tid": fe["tid"],
            })
            events.append({
                "name": f"{cat_fwd} -> {cat_bwd} pair",
                "cat":  cat_label,
                "ph":   "f", "id": flow_id,
                "ts":   be["ts"],
                "pid":  be["pid"], "tid": be["tid"],
                "bp":   "e",
            })
            flow_pair_count += 1

print(f"  fwd/bwd flow pairs emitted: {flow_pair_count:,}", file=sys.stderr)

# ----------------------------------------------------------------------------
# 6) Iter boundary instant markers + process/thread metadata
# ----------------------------------------------------------------------------
for label, t in iter_marks:
    events.append({
        "name": f"=== iter {label} (RCCL end) ===",
        "cat":  "iter_marker", "cname": "black",
        "ph":   "I", "s": "g",
        "ts":   (t - T_START) / 1000.0,
        "pid":  PID_GPU, "tid": 0,
    })

events.append({"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
               "args": {"name": f"GPU {GPU} ({TARGET_AGENT}, iters "
                                f"{FIRST_ITER}..{FIRST_ITER+N_ITERS-1}, "
                                f"{(T_END-T_START)/1e6:.2f} ms)"}})
events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
               "args": {"name": f"Host (rank {GPU})"}})

# --- Stream semantic-role classifier ---
# Match NV's nsys convention of naming streams by their role
# (computation_stream, embedding_mp, embedding_dp, sparse_prep, memcpy_stream,
# rccl_emb_ar, rccl_mlp_wgrad, etc.) by inspecting each stream's category mix.
unique_streams = sorted({e["tid"] for e in events
                         if e.get("pid") == PID_GPU and e.get("ph") == "X"})

def stream_role(sid):
    """Classify a stream's purpose from its kernel category mix."""
    cat_dur = defaultdict(float)
    for e in events:
        if e.get("pid") == PID_GPU and e.get("ph") == "X" and e["tid"] == sid:
            cat_dur[e.get("cat", "?")] += e.get("dur", 0.0)
    total = sum(cat_dur.values())
    if total < 1.0:
        return "idle_stream"
    rccl    = cat_dur.get("rccl", 0) + cat_dur.get("allreduce", 0) + cat_dur.get("emb_a2a", 0)
    mlp     = sum(cat_dur.get(c, 0) for c in
                  ("mlp_fwd", "mlp_bwd_dgrad", "mlp_bwd_wgrad",
                   "fused_fma", "interaction"))
    emb     = sum(cat_dur.get(c, 0) for c in
                  ("emb_fwd", "emb_reduce", "emb_scatter", "opt_emb",
                   "emb_other"))
    sparse  = cat_dur.get("sparse_prep", 0)
    memcpy  = (cat_dur.get("memcpy", 0) + cat_dur.get("memcpy_h2d", 0) +
               cat_dur.get("memcpy_d2h", 0) + cat_dur.get("memcpy_d2d", 0))
    memset  = cat_dur.get("memset", 0)
    loss    = cat_dur.get("loss", 0)
    other   = total - (rccl + mlp + emb + sparse + memcpy + memset + loss)

    f_rccl, f_mlp, f_emb = rccl/total, mlp/total, emb/total
    f_sparse, f_memcpy = sparse/total, memcpy/total

    # NV-style stream role names (matches nsys conventions for NV B200
    # traces so side-by-side comparison is direct):
    #   computation_stream / embedding_mp / embedding_dp / sparse_prep /
    #   memcpy_stream  +  AMD-port-only "rccl_dedicated" for the
    #   Phase-15 dedicated RCCL streams (no NV equivalent because NV
    #   runs allreduce in-line on computation_stream).
    #
    # 0. Pure-memcpy streams (H2D-dominated → data reader's placement_streams_).
    #    NV labels these "memcpy_stream" regardless of direction.
    h2d_dur = cat_dur.get("memcpy_h2d", 0)
    d2h_dur = cat_dur.get("memcpy_d2h", 0)
    if h2d_dur > 0 and (h2d_dur / total) > 0.85:
        return "memcpy_stream"
    if d2h_dur > 0 and (d2h_dur / total) > 0.85:
        return "memcpy_stream"
    if f_memcpy > 0.85:
        return "memcpy_stream"

    # 1. Pure-RCCL streams (no compute) → Phase-15 dedicated RCCL streams
    #    (rccl_emb_ar / rccl_mlp_wgrad). AMD-only; no NV equivalent.
    if f_rccl > 0.95 and f_sparse < 0.05 and f_emb < 0.05:
        return "rccl_dedicated"

    # 2. RCCL-heavy with some sparse_prep → NV's embedding_dp pattern
    #    (the alltoall-dominated embedding stream).
    if f_rccl > 0.65 and f_sparse > 0.05:
        return "embedding_dp"

    # 3. RCCL + embedding compute → NV's embedding_mp pattern
    #    (model-parallel embedding work mixed with mp alltoall).
    if f_rccl > 0.30 and f_emb > 0.15:
        return "embedding_mp"

    # 4. MLP-dominated → main computation stream
    if f_mlp > 0.50:
        return "computation_stream"

    # 5. Embedding-dominated, low RCCL → embedding_dp (data-parallel emb)
    if f_emb > 0.50 and f_rccl < 0.20:
        return "embedding_dp"

    # 6. Sparse-prep dominated
    if f_sparse > 0.50:
        return "sparse_prep"

    # 7. Memcpy/memset only
    if f_memcpy > 0.85 or (f_memcpy + cat_dur.get("memset", 0) / total) > 0.85:
        return "memcpy_stream"

    # 8. Catch-all for streams that don't cleanly match any single role.
    #    On AMD we see these when the dedicated-RCCL split leaves a
    #    "leftover" embedding stream carrying sparse_prep + emb_reduce
    #    + opt_emb. NV's closest equivalent is sparse_prep (also a
    #    catch-all for non-mainline embedding work).
    if f_sparse > 0.30:
        return "sparse_prep"
    if f_emb > 0.20:
        return "embedding_dp"
    if f_mlp > 0.20:
        return "computation_stream"
    return "other"

# NV-style sort_index per role so similar streams group visually.
# Order matches what NVIDIA nsys shows for a GPU 0 section so
# side-by-side AMD/NV comparison reads top-to-bottom.
ROLE_SORT = {
    "computation_stream":  10,
    "embedding_mp":        20,
    "embedding_dp":        30,
    "sparse_prep":         40,
    "rccl_dedicated":      45,  # AMD-port-only (Phase 15); no NV equivalent
    "memcpy_stream":       50,
    "other":               80,
    "idle_stream":         99,
}

print("  --- per-stream role classification ---", file=sys.stderr)
stream_role_by_tid = {}
for sid in unique_streams:
    role = stream_role(sid)
    stream_role_by_tid[sid] = role
    name = f"stream {sid} ({role})"
    print(f"    tid={sid}  role={role}", file=sys.stderr)
    events.append({"name": "thread_name", "ph": "M", "pid": PID_GPU, "tid": sid,
                   "args": {"name": name}})
    events.append({"name": "thread_sort_index", "ph": "M", "pid": PID_GPU, "tid": sid,
                   "args": {"sort_index": ROLE_SORT.get(role, 90)}})

# ----------------------------------------------------------------------------
# Phase-19b enhancement: RCCL operation-type tagging
#
# RCCL kernels on AMD ROCm 7.2 all share one generic name
# (ncclDevKernel_Generic_1) so we can't tell allreduce / allgather /
# alltoall apart from the kernel name alone.  Two complementary fixes:
#
#   (a) Stream-role heuristic: kernels on a rccl_dedicated stream
#       (Phase-15 dedicated rccl_emb_ar / rccl_mlp_wgrad lanes) are
#       almost certainly ncclAllReduce; kernels on the embedding_a2a
#       stream are the all-to-all send/recv collectives; kernels on the
#       sparse_prep / embedding_mp streams that match the rccl category
#       are ncclAllGather (swizzle-key all-gather).  Relabel each
#       [rccl] event in-place by stream role.
#
#   (b) Per-window summary: parse rccl_api_trace.csv (rocprofv3
#       --rccl-trace output) and emit a single instant-event annotation
#       with the per-operation-type counts in the trace window
#       (so the user can confirm e.g. "5 iters: 90 allreduces, 16
#       allgathers, 2756 send/recv = ~167 alltoall groups").
# ----------------------------------------------------------------------------
RCCL_ROLE_OP = {
    # rccl_dedicated lanes carry the Phase-15-isolated AllReduce ops
    # (rccl_emb_ar / rccl_mlp_wgrad).
    "rccl_dedicated":  "rccl_allreduce",
    # embedding_a2a / sparse_prep / embedding_mp streams all carry
    # send/recv pairs that implement the embedding alltoall (RCCL's
    # alltoall is built from grouped ncclSend+ncclRecv collectives, not
    # an ncclAllToAll primitive). DLRM-DCNv2 doesn't use ncclAllGather
    # in steady state (RCCL API summary confirms 0 allgather calls).
    "embedding_a2a":   "rccl_alltoall",
    "sparse_prep":     "rccl_alltoall",
    "embedding_mp":    "rccl_alltoall",
}

# (a) relabel each rccl event by its stream role
rccl_relabel_counts = defaultdict(int)
for e in events:
    if e.get("cat") == "rccl" and e.get("ph") == "X":
        role = stream_role_by_tid.get(e.get("tid"), "")
        op = RCCL_ROLE_OP.get(role, "rccl_collective")
        e["cat"] = op
        # rewrite display name with operation type prefix
        if e["name"].startswith("[rccl] "):
            e["name"] = f"[{op}] " + e["name"][len("[rccl] "):]
        # color hint
        e["cname"] = {"rccl_allreduce": "rail_response",
                      "rccl_alltoall":  "rail_response",
                      "rccl_allgather": "rail_response",
                      "rccl_collective": "rail_response"}[op]
        rccl_relabel_counts[op] += 1

if rccl_relabel_counts:
    print(f"  --- RCCL operation-type tagging (by stream role) ---",
          file=sys.stderr)
    for op, n in sorted(rccl_relabel_counts.items(), key=lambda x: -x[1]):
        print(f"    {n:4d}  -> {op}", file=sys.stderr)

# (b) per-window RCCL API summary from rocprofv3 --rccl-trace
if os.path.exists(RCCL_CSV):
    print(f"Loading {RCCL_CSV} (RCCL API summary) ...", file=sys.stderr)
    rccl_op_counts = defaultdict(int)
    n_rccl_in_window = 0
    with open(RCCL_CSV) as f:
        for row in csv.DictReader(f):
            fn = row.get("Function", "").strip('"')
            try:
                start = int(row["Start_Timestamp"])
            except (ValueError, KeyError):
                continue
            if start < T_START or start >= T_END:
                continue
            # only count collective ops, skip util calls
            if fn.startswith("nccl") and fn not in (
                    "ncclCommGetAsyncError", "ncclCommCount",
                    "ncclGetVersion", "ncclCommDestroy",
                    "ncclGetUniqueId", "ncclCommInitAll"):
                rccl_op_counts[fn] += 1
                n_rccl_in_window += 1
    if rccl_op_counts:
        print(f"  RCCL API calls in window: {n_rccl_in_window}", file=sys.stderr)
        for fn, n in sorted(rccl_op_counts.items(), key=lambda x: -x[1]):
            print(f"    {n:5d}  {fn}", file=sys.stderr)
        # number of all-to-all groups = number of ncclGroupStart calls
        # (each group bracket = one logical alltoall collective)
        n_a2a_groups = rccl_op_counts.get("ncclGroupStart", 0)
        n_send = rccl_op_counts.get("ncclSend", 0)
        # emit a single instant marker at trace start with the summary
        summary = (
            f"RCCL API in window ({N_ITERS} iters): "
            f"{rccl_op_counts.get('ncclAllReduce', 0)} allreduce, "
            f"{rccl_op_counts.get('ncclAllGather', 0)} allgather, "
            f"~{n_a2a_groups} alltoall groups "
            f"({n_send} send/recv pairs)"
        )
        events.append({
            "name": summary,
            "cat": "rccl_summary",
            "ph": "I",   # instant event
            "ts": 0.0,
            "pid": PID_GPU,
            "tid": 0,
            "s": "p",    # process-wide
            "args": dict(rccl_op_counts),
        })
else:
    print(f"  (no {RCCL_CSV} present -- run rocprofv3 with "
          f"--rccl-trace to populate the RCCL op-type summary)",
          file=sys.stderr)

for tid, lane in host_tid_to_lane.items():
    events.append({"name": "thread_name", "ph": "M", "pid": PID_HOST, "tid": lane,
                   "args": {"name": f"host_thread {tid}"}})
    events.append({"name": "thread_sort_index", "ph": "M", "pid": PID_HOST, "tid": lane,
                   "args": {"sort_index": 0}})

# ----------------------------------------------------------------------------
# Write JSON
# ----------------------------------------------------------------------------
print(f"\nTotal events: {len(events):,}", file=sys.stderr)
print(f"Writing {OUT} ...", file=sys.stderr)
with open(OUT, "w") as f:
    json.dump({
        "traceEvents": events,
        "displayTimeUnit": "ms",
        "metadata": {
            "src": args.trace_prefix, "gpu": GPU, "agent": TARGET_AGENT,
            "iter_window": [FIRST_ITER, FIRST_ITER + N_ITERS - 1],
            "T_START_ns": T_START, "window_ms": (T_END - T_START) / 1e6,
            "categories": dict(cat_count),
            "launch_arrows": launch_pairs,
            "fwd_bwd_arrows": flow_pair_count,
            "iter_detection": f"RCCL gap > {args.iter_gap_us} us",
        }
    }, f)
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT} ({sz:.2f} MB)", file=sys.stderr)
