#!/usr/bin/env python3
"""
Phase 20.10 (Path A, 2026-05-20): overlay vendor kernel names from a
ONE-SHOT rocprofv3 cold-pass capture onto a low-overhead HCTR native
trace (HCTR_NATIVE_TRACE=1 output).

Why this exists
---------------
rocprofv3 --kernel-trace is the only way to learn the exact GPU kernel
symbol that fires at a given HCTR phase (especially hipBLASLt Tensile
kernels and RCCL collectives, whose names depend on shape and op type).
But running every production trace under rocprofv3 inflates iter time
by +193%..+370% because rocprofv3 is not HIP-Graph-aware and re-tracks
every replayed node as a fresh dispatch.

This script lets you pay the rocprofv3 tax ONCE per build to learn the
mapping, then reuse it on every subsequent production-near native
trace at +3% overhead.

How the join works
------------------
The HCTR phase name (set at GpuPhase registration) is stable across all
runs of a given build, because it is derived from C++ call-site
identity. So the mapping
    phase_name  ->  (kernel_symbol, gemm_shape, vgpr, sgpr, lds, grid)
can be precomputed from a single cold-pass iter and applied to any
production native trace produced by the same binary.

The cold-pass run MUST have:
  * HCTR_NATIVE_TRACE=0     (no native events, no overhead concern)
  * HCTR_ROCTX=1            (ROCTX ranges = phase names visible to rocprofv3)
  * rocprofv3 --kernel-trace --hip-trace --marker-trace ...

The native (hot) run MUST have:
  * HCTR_NATIVE_TRACE=1     (produces hctr_native_trace_rank${R}.json)
  * NO rocprofv3 attached   (otherwise you defeat the point)

Inputs
------
  --native-json   PATH    HCTR native JSON (one rank)
  --cold-prefix   PATH    rocprofv3 cold-pass output prefix (matches the
                          path used by trace_and_convert.sh; we read
                          ${prefix}_kernel_trace.csv,
                          ${prefix}_marker_api_trace.csv, etc.)
  --gpu           INT     GPU index inside the cold-pass run (default 0)
  --out           PATH    Output Perfetto JSON (default: <native>.overlay.json)
  --cold-iter     INT     Which iter of the cold pass to use to build the
                          phase->kernel map (default: 5; skips warmup)

Output
------
A Perfetto-loadable JSON with each native event's args augmented:
  args.kernel              <- demangled (or Tensile-decoded) kernel name
  args.kernel_role         <- "hipBLASLt_gemm_fwd" / "rccl_alltoall" / ...
  args.macro_tile          <- "MT128x176x128" (hipBLASLt only)
  args.mfma                <- "MI16x16x1" (hipBLASLt only)
  args.dtype               <- "HHS" (hipBLASLt only)
  args.vgpr_per_thread     <- from cold-pass kernel CSV
  args.sgpr_count          <- from cold-pass kernel CSV
  args.lds_bytes           <- from cold-pass kernel CSV
  args.grid_workgroups     <- from cold-pass kernel CSV
  args.workgroup_size      <- from cold-pass kernel CSV
And the slice DISPLAY name is rewritten to
  "[<cat>] <demangled-or-decoded-short-name>"
so Perfetto's lane view shows the real kernel symbol where rocprofv3
would have shown it.

Phases that already have args.kernel set (Path D in-tree resolution)
are LEFT ALONE; the overlay only fills in missing fields. This makes
the two paths additive and order-independent.

Usage
-----
  # 1. one-shot cold pass (slow, +370% overhead, run for 25 iters)
  HCTR_ROCTX=1 trace_and_convert.sh ...
  # produces /home/chcai/rps_out/iter15_20_kernel_trace.csv etc.

  # 2. production native trace (fast, +3-30% overhead)
  HCTR_NATIVE_TRACE=1 HCTR_NATIVE_TRACE_DETAIL=2 \\
      HCTR_NATIVE_TRACE_DIR=/home/chcai/native_out \\
      run_b200_match.sh ...
  # produces /home/chcai/native_out/hctr_native_trace_rank0.json

  # 3. overlay (default: explodes [graph] network into per-kernel slices)
  python3 native_trace_overlay.py \\
      --native-json /home/chcai/native_out/hctr_native_trace_rank0.json \\
      --cold-prefix /home/chcai/rps_out/iter15_20 \\
      --gpu 0 \\
      --out /home/chcai/native_out/overlay_rank0.json

  # Legacy single-slice graph bucket:
  python3 native_trace_overlay.py ... --no-explode-graph
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

# ---------------------------------------------------------------------------
# Tensile (hipBLASLt) kernel-name parser. Same regex set as the existing
# rocprofv3_to_perfetto_annotated.py converter -- factored out here so the
# overlay can decode shapes without depending on that script.
# ---------------------------------------------------------------------------
_RE_HIPBLASLT_MT = re.compile(r"_MT(\d+)x(\d+)x(\d+)_")
_RE_HIPBLASLT_MI = re.compile(r"_MI(\d+)x(\d+)x(\d+)")
_RE_HIPBLASLT_LAYOUT = re.compile(r"^Cijk_(\w{4})_(\w{4})_")
_RE_HIPBLASLT_DTYPE = re.compile(r"^Cijk_\w{4}_\w{4}_([A-Z]{2,4})_([A-Z]{1,4})_")


def parse_hipblaslt_kernel_name(name):
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


def short_kernel_label(kernel_name, gemm_info):
    """Return a compact lane label for Perfetto.

    For Tensile names we collapse to "hipBLASLt MT128x176x128 (HHS)".
    For other names we strip template parameters, function arguments,
    and namespace prefixes so heavily-templated HCTR/rocPRIM symbols
    render as a single readable identifier in the Perfetto lane.
    """
    if gemm_info.get("macro_tile"):
        out = f"hipBLASLt MT{gemm_info['macro_tile']}"
        if gemm_info.get("dtype"):
            out += f" ({gemm_info['dtype']})"
        return out
    if not kernel_name:
        return ""
    s = kernel_name
    # Remove the GCC-internal "(anonymous namespace)" marker that the
    # demangler injects around unnamed namespaces; otherwise it gets
    # caught by the balanced-paren strip below and breaks the function
    # arg detection.
    s = s.replace("(anonymous namespace)::", "").replace("(anonymous namespace)", "")

    def _strip_balanced(text, opener, closer):
        """Iteratively remove the outermost <...> / (...) blocks."""
        while opener in text:
            depth = 0
            out = []
            removed_any = False
            for ch in text:
                if ch == opener:
                    depth += 1
                    if depth == 1:
                        removed_any = True
                        continue
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        continue
                if depth == 0:
                    out.append(ch)
            text = "".join(out)
            if not removed_any:
                break
        return text

    # Strip template parameters first (balanced <...>), then function
    # argument lists (balanced (...)).
    s = _strip_balanced(s, "<", ">")
    s = _strip_balanced(s, "(", ")")
    # Drop a leading "void "/"static " return-type prefix.
    for ret in ("void ", "static "):
        if s.startswith(ret):
            s = s[len(ret):]
    # Strip namespace prefixes -- show only the leaf symbol.
    if "::" in s:
        s = s.split("::")[-1]
    return s.strip()[:80]


def is_graph_explode_phase(name):
    """True if this native slice is the captured-network HIP graph bucket."""
    for k in normalize_phase_name(name):
        if k in _GRAPH_EXPLODE_PHASES:
            return True
    return False


def kernel_info_from_cold(k):
    """Build overlay args + display fields for one cold-pass kernel row."""
    gemm = parse_hipblaslt_kernel_name(k["name"])
    info = {
        "kernel": k["name"],
        "kernel_short": short_kernel_label(k["name"], gemm),
        "duration_us_cold": round((k["end"] - k["start"]) / 1000.0, 3),
        "vgpr_per_thread": k["vgpr"],
        "sgpr_count": k["sgpr"],
        "lds_bytes": k["lds"],
        "grid_workgroups":
            f"{k['gx_blocks']}x{k['gy_blocks']}x{k['gz_blocks']}",
        "workgroup_size": f"{k['wgx']}x{k['wgy']}x{k['wgz']}",
    }
    info.update(gemm)
    return info


def pick_graph_kernel_list(phase_kernel_lists, min_kernels=2):
    """Ordered cold-pass kernels inside the captured network graph."""
    for key in ("[graph] network", "graph_network"):
        klist = phase_kernel_lists.get(key)
        if klist and len(klist) >= min_kernels:
            return sorted(klist, key=lambda k: k["start"])
    return None


def pick_phase_kernel_list(phase_kernel_lists, phase_name, min_kernels=2):
    """Cold-pass kernels attributed to a native GpuPhase name."""
    if is_graph_explode_phase(phase_name):
        return pick_graph_kernel_list(phase_kernel_lists, min_kernels)
    best = None
    for key in normalize_phase_name(phase_name):
        klist = phase_kernel_lists.get(key)
        if klist and len(klist) >= min_kernels:
            ordered = sorted(klist, key=lambda k: k["start"])
            if best is None or len(ordered) > len(best):
                best = ordered
    return best


def phase_name_prefix(phase_name):
    """Keep [cat] prefix for exploded slice labels."""
    if phase_name.startswith("[") and "] " in phase_name:
        return phase_name.split("] ", 1)[0] + "] "
    return ""


# Perfetto tid strings for multi-kernel explosion targets.
_EXPLODE_TID_PREFETCH = "prefetch"
_EXPLODE_TID_DEFAULTDP = "defaultdp"
_EXPLODE_TID_DEFAULTMP = "defaultmp"


def explode_tid_enabled(tid, explode_graph, explode_prefetch, explode_defaultdp,
                        explode_defaultmp):
    if explode_graph:
        pass  # graph uses phase name, not tid
    if explode_prefetch and tid == _EXPLODE_TID_PREFETCH:
        return True
    if explode_defaultdp and tid == _EXPLODE_TID_DEFAULTDP:
        return True
    if explode_defaultmp and tid == _EXPLODE_TID_DEFAULTMP:
        return True
    return False


def should_explode_event(name, tid, phase_kernel_lists, explode_graph,
                         explode_prefetch, explode_defaultdp, explode_defaultmp,
                         min_kernels):
    """Whether a native slice will be fan-out into per-kernel children."""
    if explode_graph and is_graph_explode_phase(name):
        return pick_phase_kernel_list(
            phase_kernel_lists, name, min_kernels) is not None
    if explode_tid_enabled(tid, False, explode_prefetch, explode_defaultdp,
                           explode_defaultmp):
        return pick_phase_kernel_list(phase_kernel_lists, name, min_kernels) is not None
    return False


# Keys copied from parent overlay args that are wrong/misleading on children.
_PARENT_ARGS_DROP = frozenset((
    "kernel", "kernel_short", "kernel_secondary", "n_kernels_in_phase",
    "duration_us", "macro_tile", "macro_tile_m", "macro_tile_n", "macro_tile_k",
    "mfma", "mfma_m", "mfma_n", "mfma_k", "dtype", "dtype_compute",
    "layout_a", "layout_b", "vgpr_per_thread", "sgpr_count", "lds_bytes",
    "grid_workgroups", "workgroup_size",
))


# Phase names the overlay should leave alone (no kernel-name attribution
# makes sense for these). Anchored at the start of the slice name.
_PHASE_NAME_SKIPS = (
    "iter_",                 # Iter envelopes wrap every kernel in an iter
    "[host_api] ",           # ScopedHostTimer events have no GPU kernel
    "[sync] sync_back",      # Empty wait-barrier scheduleable, no kernel
)

# Native GpuPhase names (and cold-pass ROCTX aliases) whose captured HIP
# graph should be exploded into one Perfetto slice per kernel.
_GRAPH_EXPLODE_PHASES = frozenset(("[graph] network", "graph_network"))


def normalize_phase_name(s):
    """Generate the candidate key set for matching a native phase name
    against the cold-pass ROCTX ranges.

    The ROCTX strings emitted at C++ call sites are not always the
    exact same string as the native emitter's GpuPhase name. Common
    patterns we need to bridge:
      native "[graph] network"  <->  rocprofv3 "graph_network"
      native "[cat] sub_name"   <->  rocprofv3 "[cat] sub_name"
      native "[cat] sub_name"   <->  rocprofv3 "cat_sub_name"
      native "sub_name"         <->  rocprofv3 "[cat] sub_name"
    """
    keys = [s]
    # "[cat] name" -> "cat_name"  (drop brackets, replace " " with "_")
    if s.startswith("[") and "] " in s:
        cat = s[1:s.index("]")]
        rest = s.split("] ", 1)[1]
        keys.append(f"{cat}_{rest.replace(' ', '_')}")
        keys.append(rest)
    # "cat_name" -> "[cat] name"
    if "_" in s and not s.startswith("["):
        cat, _, rest = s.partition("_")
        keys.append(f"[{cat}] {rest.replace('_', ' ')}")
        keys.append(f"[{cat}] {rest}")
    return keys


# ---------------------------------------------------------------------------
# Cold-pass loading: build phase_name -> kernel info from rocprofv3 CSVs
#
# Strategy: walk the marker_api_trace.csv (ROCTX ranges = phase names)
# and find, for each range, the kernel(s) that dispatched while the
# range was active on the same TID/agent. A range typically wraps
# exactly one HCTR kernel; for fused MLP wrappers (DETAIL=1) it can
# wrap a small kernel group, in which case we pick the longest kernel
# as the "primary" and stash the rest in args.kernel_secondary.
# ---------------------------------------------------------------------------
def load_cold_pass(prefix, gpu_idx, cold_iter):
    kt_csv = prefix + "_kernel_trace.csv"
    mk_csv = prefix + "_marker_api_trace.csv"
    ag_csv = prefix + "_agent_info.csv"
    hp_csv = prefix + "_hip_api_trace.csv"

    for p in (kt_csv, ag_csv):
        if not os.path.exists(p):
            sys.exit(f"ERROR: missing cold-pass CSV: {p}")

    # Resolve TARGET_AGENT for GPU index.
    gpu_agents = []
    with open(ag_csv) as f:
        for row in csv.DictReader(f):
            if row.get("Agent_Type") == "GPU":
                gpu_agents.append(int(row["Logical_Node_Id"]))
    gpu_agents.sort()
    if gpu_idx >= len(gpu_agents):
        sys.exit(f"ERROR: gpu {gpu_idx} out of range (have {len(gpu_agents)} GPUs)")
    target_agent = f"Agent {gpu_agents[gpu_idx]}"

    # Load kernel events on target agent, sorted by start.
    kernels = []  # (start_ns, end_ns, tid, name, corr, vgpr, sgpr, lds, wgx, wgy, wgz, gx, gy, gz)
    with open(kt_csv) as f:
        for row in csv.DictReader(f):
            if row.get("Agent_Id") != target_agent:
                continue
            try:
                s = int(row["Start_Timestamp"])
                e = int(row["End_Timestamp"])
            except (ValueError, KeyError):
                continue
            wgx = int(row.get("Workgroup_Size_X", 0) or 0)
            wgy = int(row.get("Workgroup_Size_Y", 0) or 0)
            wgz = int(row.get("Workgroup_Size_Z", 0) or 0)
            gx = int(row.get("Grid_Size_X", 0) or 0)
            gy = int(row.get("Grid_Size_Y", 0) or 0)
            gz = int(row.get("Grid_Size_Z", 0) or 0)
            kernels.append({
                "start": s,
                "end": e,
                "tid": int(row.get("Thread_Id", 0) or 0),
                "name": row.get("Kernel_Name", ""),
                "corr": int(row.get("Correlation_Id", 0) or 0),
                "sid": int(row.get("Stream_Id", 0) or 0),
                "vgpr": int(row.get("VGPR_Count", 0) or 0),
                "sgpr": int(row.get("SGPR_Count", 0) or 0),
                "lds": int(row.get("LDS_Block_Size", 0) or 0),
                "wgx": wgx, "wgy": wgy, "wgz": wgz,
                "gx_blocks": (gx // wgx) if wgx else gx,
                "gy_blocks": (gy // wgy) if wgy else gy,
                "gz_blocks": (gz // wgz) if wgz else gz,
            })
    kernels.sort(key=lambda k: k["start"])

    # Walk hip_api to learn:
    #   (a) the dispatch tid for each Correlation_Id (for the GPU-time
    #       bracket-fallback path), and
    #   (b) the (start_ns, end_ns, tid, correlation_id) tuple for every
    #       HIP API call, so we can attribute kernels to ROCTX ranges
    #       by host-side API correlation rather than GPU-side time.
    #
    # Why correlation matters: short host-side ROCTX ranges (e.g.
    # mlp_wgrad_allreduce, ~0.25 ms) wrap *enqueue* calls into RCCL.
    # The corresponding GPU kernels fire on the dedicated RCCL stream
    # MUCH later (post-graph-launch), so pure GPU-time bracket misses
    # them. By keying on correlation_id we follow the host->device
    # link that rocprofv3 already provides.
    corr_to_dispatch_tid = {}
    hip_api_events = []  # list of (start_ns, end_ns, tid, corr)
    if os.path.exists(hp_csv):
        with open(hp_csv) as f:
            for row in csv.DictReader(f):
                try:
                    corr = int(row.get("Correlation_Id", 0) or 0)
                    tid = int(row.get("Thread_Id", 0) or 0)
                    s = int(row["Start_Timestamp"])
                    e = int(row["End_Timestamp"])
                except (ValueError, KeyError):
                    continue
                if corr and tid and corr not in corr_to_dispatch_tid:
                    corr_to_dispatch_tid[corr] = tid
                if corr:
                    hip_api_events.append((s, e, tid, corr))
    hip_api_events.sort()

    # Load marker ranges (ROCTX phase names) -- one entry per push/pop pair.
    # Each entry is (start_ns, end_ns, tid, phase_name).
    ranges = []
    if os.path.exists(mk_csv):
        with open(mk_csv) as f:
            for row in csv.DictReader(f):
                domain = row.get("Domain", "")
                if domain not in ("MARKER_CORE_RANGE_API", "MARKER_CORE_API"):
                    continue
                name = row.get("Function", "")
                # Skip ROCTX internals and iter-only markers (we want
                # phase ranges that wrap kernel work).
                if not name or name.startswith("roctx"):
                    continue
                try:
                    s = int(row["Start_Timestamp"])
                    e = int(row["End_Timestamp"])
                    tid = int(row.get("Thread_Id", 0) or 0)
                except (ValueError, KeyError):
                    continue
                ranges.append((s, e, tid, name))
    ranges.sort()

    # Find iter window: iter boundaries are the START of each phase
    # named "iter_<N>" (these are emitted by Model::train). Use cold_iter
    # to pick which iter's mapping we trust.
    iter_starts = {}  # iter_idx -> start_ns
    iter_ends = {}    # iter_idx -> end_ns
    for s, e, _tid, nm in ranges:
        if nm.startswith("iter_"):
            try:
                idx = int(nm.split("_", 1)[1])
            except ValueError:
                continue
            iter_starts.setdefault(idx, s)
            # The "iter_N" range covers the entire iter, so end is the
            # range end; if multiple instances (rare) take the longest.
            iter_ends[idx] = max(iter_ends.get(idx, 0), e)

    if cold_iter in iter_starts:
        win_start = iter_starts[cold_iter]
        win_end = iter_ends[cold_iter]
    elif iter_starts:
        chosen = sorted(iter_starts)[len(iter_starts) // 2]
        sys.stderr.write(
            f"  WARNING: cold_iter={cold_iter} not found in cold pass; "
            f"falling back to iter {chosen}\n")
        win_start = iter_starts[chosen]
        win_end = iter_ends[chosen]
    else:
        # No iter_N marker emitted -- fall back to the entire trace window.
        if not kernels:
            sys.exit("ERROR: cold pass has no kernels on target agent")
        win_start = kernels[0]["start"]
        win_end = kernels[-1]["end"]

    sys.stderr.write(
        f"  cold-pass iter window: [{win_start}, {win_end}] "
        f"({(win_end - win_start)/1e6:.3f} ms)\n")

    # Build the mapping: phase_name -> [list of kernel-info dicts that
    # fired while the range was active in the chosen iter window]. We
    # require that the kernel start lies inside the range AND inside the
    # iter window.
    phase_to_kernels = defaultdict(list)
    in_window_ranges = [(s, e, tid, nm) for (s, e, tid, nm) in ranges
                        if not (e <= win_start or s >= win_end)]
    sys.stderr.write(
        f"  cold-pass phase ranges in iter window: {len(in_window_ranges)}\n")

    # Two-stage attribution per ROCTX range:
    #
    # Stage 1 (correlation-driven, preferred). Find all HIP API events
    # whose host-side timing falls inside the ROCTX range AND that
    # were emitted from the same tid as the range. Collect their
    # correlation_ids; attribute every kernel with a matching corr_id
    # to this phase. This correctly handles async cases like RCCL on
    # dedicated streams (where kernel timestamps fall LATER than the
    # host range) and is robust against the iter envelope range
    # (which is short and would otherwise miss its async kernels).
    #
    # Stage 2 (GPU-time bracket fallback). For ranges whose host span
    # contains no useful HIP API correlations OR where the API set
    # over-includes (e.g. long-lived ranges like graph_network), also
    # attribute any kernel whose GPU start falls inside the range
    # (filtered by stream / agent already). The union of the two
    # attributions is what we use.
    #
    # Multiple ranges may attribute the same kernel; we keep all
    # mappings since downstream picks the *longest* kernel as primary
    # per phase, and any reasonable phase->kernel link is informative.
    import bisect

    # Pre-index hip_api_events by start for fast bracketing.
    api_starts = [t[0] for t in hip_api_events]
    # Pre-index kernels by Correlation_Id for fast lookup.
    kernels_by_corr = defaultdict(list)
    for k in kernels:
        if k["corr"]:
            kernels_by_corr[k["corr"]].append(k)

    # Stage 2 (GPU-time bracket) is a fallback for phases that wrap
    # hipGraphLaunch -- the kernels INSIDE the graph don't always have
    # matching correlation_ids in hip_api_trace, so Stage 1 may miss
    # them. We only apply Stage 2 when:
    #   - The phase has 0 hits from Stage 1, AND
    #   - The range duration is below a "short" threshold (so GPU-time
    #     bracketing is reliably a single-phase attribution; for
    #     longer ranges like cold-pass [emb_fwd] ebc_dp ~10 ms, Stage 2
    #     would over-attribute kernels from concurrent streams).
    SHORT_RANGE_NS = 2_000_000  # 2 ms

    starts = [k["start"] for k in kernels]
    for s, e, tid, name in in_window_ranges:
        # ---- Stage 1: correlation-id matching (filtered by range tid) ----
        # We MUST filter API events by the ROCTX range's owning tid,
        # otherwise on medium-length ranges (e.g. cold-pass
        # [emb_fwd] ebc_dp ~10 ms) we vacuum up unrelated API events
        # from other threads (pipeline scheduleables, RCCL workers,
        # etc.). The kernel itself may be dispatched on a worker
        # thread by HIP runtime, but the originating hipLaunchKernel
        # call always sits on the C++-scope-owning thread.
        lo_api = bisect.bisect_left(api_starts, s)
        hi_api = bisect.bisect_left(api_starts, e)
        corr_set = set()
        for (_as, _ae, atid, acorr) in hip_api_events[lo_api:hi_api]:
            if tid and atid and atid != tid:
                continue
            corr_set.add(acorr)
        stage1_hits = 0
        for corr in corr_set:
            for k in kernels_by_corr.get(corr, ()):
                phase_to_kernels[name].append(k)
                stage1_hits += 1

        # ---- Stage 2: GPU-time bracket fallback (small ranges only) ----
        # Active only when Stage 1 found nothing AND the range is
        # short enough that GPU-time bracketing is unambiguous (no
        # concurrent-stream contamination). Catches hipGraphLaunch
        # phases where graph-internal kernels lack hip_api correls.
        if stage1_hits == 0 and (e - s) <= SHORT_RANGE_NS:
            lo = bisect.bisect_left(starts, s)
            hi = bisect.bisect_left(starts, e)
            for k in kernels[lo:hi]:
                phase_to_kernels[name].append(k)

    # De-duplicate within each phase (a kernel can be added twice via
    # both stages).
    for name in phase_to_kernels:
        seen = set()
        deduped = []
        for k in phase_to_kernels[name]:
            key = (k["start"], k["corr"], k["name"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(k)
        phase_to_kernels[name] = deduped

    # Pick the "primary" kernel per phase (longest by GPU duration) and
    # record metadata. For phases that wrap multiple kernels, remember
    # the others as kernel_secondary[].
    mapping = {}
    for phase_name, klist in phase_to_kernels.items():
        if not klist:
            continue
        klist_sorted = sorted(klist, key=lambda k: -(k["end"] - k["start"]))
        primary = klist_sorted[0]
        secondaries = klist_sorted[1:]
        gemm = parse_hipblaslt_kernel_name(primary["name"])
        entry = {
            "kernel": primary["name"],
            "kernel_short": short_kernel_label(primary["name"], gemm),
            "n_kernels_in_phase": len(klist),
            "duration_us": round((primary["end"] - primary["start"]) / 1000.0, 3),
            "vgpr_per_thread": primary["vgpr"],
            "sgpr_count": primary["sgpr"],
            "lds_bytes": primary["lds"],
            "grid_workgroups":
                f"{primary['gx_blocks']}x{primary['gy_blocks']}x{primary['gz_blocks']}",
            "workgroup_size":
                f"{primary['wgx']}x{primary['wgy']}x{primary['wgz']}",
        }
        entry.update(gemm)
        # kernel_secondary was a truncated list of "other" kernels in the
        # same ROCTX bracket. Stage-1 correlation often pulls in unrelated
        # async kernels (e.g. RCCL on another stream), so we omit it for
        # multi-kernel phases; per-kernel explosion is the accurate view.
        if len(klist) == 1:
            entry["kernel_secondary"] = []
        mapping[phase_name] = entry

    sys.stderr.write(
        f"  cold-pass phase->kernel mappings learned: {len(mapping)}\n")
    return mapping, dict(phase_to_kernels)


# ---------------------------------------------------------------------------
# Multi-kernel explosion: native aggregate slice -> N kernel slices
# ---------------------------------------------------------------------------
def explode_multi_kernel_phases(events, phase_kernel_lists, explode_graph=True,
                                explode_prefetch=True, explode_defaultdp=True,
                                explode_defaultmp=True, min_kernels=2):
    """Fan-out native slices that wrap many kernels (graph, prefetch, …).

    Children stay on the parent's Perfetto tid (e.g. ``default`` for
    ``[graph] network``, ``prefetch`` for sparse_prep). Kernel order and
    relative GPU durations come from the cold pass; wall times are scaled
    to the native tracer's measured phase duration (production timing).
    """
    out = []
    exploded_children = 0
    removed_parents = 0
    by_tid = defaultdict(int)

    for ev in events:
        if ev.get("ph") != "X":
            out.append(ev)
            continue

        name = ev.get("name", "")
        tid = ev.get("tid", "default")
        if explode_graph and is_graph_explode_phase(name):
            pass
        elif explode_tid_enabled(tid, False, explode_prefetch, explode_defaultdp,
                                 explode_defaultmp):
            pass
        else:
            out.append(ev)
            continue

        klist = pick_phase_kernel_list(phase_kernel_lists, name, min_kernels)
        if not klist:
            out.append(ev)
            continue

        cold_durs_us = [(k["end"] - k["start"]) / 1000.0 for k in klist]
        cold_total_us = sum(cold_durs_us)
        if cold_total_us <= 0:
            out.append(ev)
            continue

        parent_ts = float(ev["ts"])
        parent_dur = float(ev["dur"])
        parent_args = {
            k: v for k, v in ev.get("args", {}).items()
            if k not in _PARENT_ARGS_DROP
        }
        parent_pid = ev.get("pid", 0)
        parent_cat = ev.get("cat", "phase")
        prefix = phase_name_prefix(name)
        scale = parent_dur / cold_total_us

        cumulative = 0.0
        children = []
        for idx, (k, raw_us) in enumerate(zip(klist, cold_durs_us)):
            info = kernel_info_from_cold(k)
            ks = info.get("kernel_short") or info.get("kernel", "")
            child_dur = raw_us * scale
            children.append({
                "name": f"{prefix}{ks}" if ks else f"{prefix}kernel_{idx}",
                "cat": parent_cat,
                "ph": "X",
                "ts": parent_ts + cumulative,
                "dur": child_dur,
                "pid": parent_pid,
                "tid": tid,
                "args": {
                    **parent_args,
                    **info,
                    "kernel_idx": idx,
                    "n_kernels_in_phase": len(klist),
                    "phase_exploded": True,
                    "phase_parent": name,
                },
            })
            cumulative += child_dur

        if children:
            end_ts = parent_ts + parent_dur
            last = children[-1]
            drift = end_ts - (last["ts"] + last["dur"])
            if abs(drift) > 1e-6:
                last["dur"] = max(0.0, last["dur"] + drift)

        out.extend(children)
        exploded_children += len(children)
        removed_parents += 1
        by_tid[tid] += len(children)

    if exploded_children:
        tid_summary = ", ".join(f"{t}={n}" for t, n in sorted(by_tid.items()))
        sys.stderr.write(
            f"  phase explode: replaced {removed_parents} parent slice(s) with "
            f"{exploded_children} kernel slice(s) ({tid_summary})\n")
    return out, exploded_children, removed_parents


# ---------------------------------------------------------------------------
# Overlay: walk native JSON, inject kernel info into args + rewrite display
# ---------------------------------------------------------------------------
def overlay(native_json_path, mapping, out_path, phase_kernel_lists=None,
            explode_graph=True, explode_prefetch=True,
            explode_defaultdp=True, explode_defaultmp=True):
    with open(native_json_path) as f:
        doc = json.load(f)

    events = doc.get("traceEvents", [])
    sys.stderr.write(
        f"  native events: {len(events)} in {native_json_path}\n")

    hits = 0
    misses = 0
    already_set = 0
    skipped = 0
    miss_names = defaultdict(int)
    for ev in events:
        if ev.get("ph") != "X":
            continue
        name = ev.get("name", "")
        # Skip phases that have no associated GPU kernel by design
        # (iter envelopes, host-side timers). They keep their original
        # name and timing.
        if any(name.startswith(pfx) for pfx in _PHASE_NAME_SKIPS):
            skipped += 1
            continue
        args = ev.setdefault("args", {})
        # Path D in-tree resolution may have populated args.kernel
        # already; leave those slices alone.
        if "kernel" in args and args["kernel"]:
            already_set += 1
            continue

        # Try multiple key variants to bridge minor naming differences
        # between the C++ ScopedRange strings and GpuPhase strings.
        info = None
        for k in normalize_phase_name(name):
            info = mapping.get(k)
            if info is not None:
                break
        if info is None:
            misses += 1
            miss_names[name] += 1
            continue
        hits += 1

        for k, v in info.items():
            if k in args and args[k]:
                continue
            args[k] = v

        # Rewrite display to surface the kernel symbol while preserving
        # the [cat] prefix. For coarse-wrapping phases (graph_network,
        # iter envelopes if they ever reach this code path) we annotate
        # the slice with the kernel count instead of pretending it has
        # one primary kernel.
        ks = info.get("kernel_short") or info.get("kernel") or ""
        n_in_phase = info.get("n_kernels_in_phase", 1)
        # Multi-kernel phases are exploded later; keep the native phase name.
        if ks and not should_explode_event(
                name, ev.get("tid", "default"), phase_kernel_lists or {},
                explode_graph, explode_prefetch, explode_defaultdp,
                explode_defaultmp, min_kernels=2):
            if name.startswith("[") and "] " in name:
                cat_prefix = name.split("] ", 1)[0] + "] "
                if n_in_phase > 8:
                    ev["name"] = f"{cat_prefix}{n_in_phase} kernels (top: {ks})"
                else:
                    ev["name"] = cat_prefix + ks
            else:
                seg = name.split("/", 1)[0] if "/" in name else "phase"
                if n_in_phase > 8:
                    ev["name"] = f"[{seg}] {n_in_phase} kernels (top: {ks})"
                else:
                    ev["name"] = f"[{seg}] {ks}"

    sys.stderr.write(
        f"  overlay results: hits={hits} misses={misses} "
        f"already_set={already_set} skipped={skipped}\n")

    phase_exploded = 0
    phase_parents_removed = 0
    if phase_kernel_lists is not None and (
            explode_graph or explode_prefetch or explode_defaultdp
            or explode_defaultmp):
        events, phase_exploded, phase_parents_removed = explode_multi_kernel_phases(
            events, phase_kernel_lists,
            explode_graph=explode_graph,
            explode_prefetch=explode_prefetch,
            explode_defaultdp=explode_defaultdp,
            explode_defaultmp=explode_defaultmp)
        doc["traceEvents"] = events

    if misses and len(miss_names) <= 30:
        sys.stderr.write("  top miss phase names (no cold-pass mapping):\n")
        for nm, c in sorted(miss_names.items(), key=lambda x: -x[1])[:30]:
            sys.stderr.write(f"    {c:5d}  {nm}\n")

    # Annotate metadata so downstream readers know this is an overlay.
    md = doc.setdefault("metadata", {})
    md["overlay_applied"] = True
    md["overlay_hits"] = hits
    md["overlay_misses"] = misses
    md["overlay_phases_in_map"] = len(mapping)
    md["overlay_phase_exploded"] = bool(
        explode_graph or explode_prefetch or explode_defaultdp
        or explode_defaultmp)
    md["overlay_phase_kernel_slices"] = phase_exploded
    md["overlay_phase_parents_removed"] = phase_parents_removed
    md["overlay_explode_graph"] = bool(explode_graph)
    md["overlay_explode_prefetch"] = bool(explode_prefetch)
    md["overlay_explode_defaultdp"] = bool(explode_defaultdp)
    md["overlay_explode_defaultmp"] = bool(explode_defaultmp)

    with open(out_path, "w") as f:
        json.dump(doc, f)
    sz = os.path.getsize(out_path) / 1e6
    sys.stderr.write(f"  wrote {out_path} ({sz:.2f} MB)\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native-json", required=True,
                    help="Path to HCTR native trace JSON for ONE rank")
    ap.add_argument("--cold-prefix", required=True,
                    help="rocprofv3 cold-pass output prefix (no _kernel_trace.csv)")
    ap.add_argument("--gpu", type=int, default=0,
                    help="GPU index inside the cold-pass run (default 0)")
    ap.add_argument("--cold-iter", type=int, default=5,
                    help="Which iter of the cold pass to harvest "
                         "(default 5; skips warmup)")
    ap.add_argument("--out", default=None,
                    help="Output JSON path (default: <native-json>.overlay.json)")
    ap.add_argument("--dump-map", default=None,
                    help="Also write the phase->kernel mapping as JSON "
                         "(useful for caching / cross-run reuse)")
    ap.add_argument("--no-explode-graph", action="store_true",
                    help="Keep a single [graph] network aggregate slice")
    ap.add_argument("--no-explode-prefetch", action="store_true",
                    help="Keep aggregate prefetch sparse_prep slices")
    ap.add_argument("--no-explode-defaultdp", action="store_true",
                    help="Keep aggregate defaultdp sparse_prep slices")
    ap.add_argument("--no-explode-defaultmp", action="store_true",
                    help="Keep aggregate defaultmp emb_a2a slices")
    ap.add_argument("--no-explode-phases", action="store_true",
                    help="Disable all per-kernel phase explosion")
    args = ap.parse_args()

    out_path = args.out
    if out_path is None:
        if args.native_json.endswith(".json"):
            out_path = args.native_json[:-5] + ".overlay.json"
        else:
            out_path = args.native_json + ".overlay.json"

    sys.stderr.write(
        f"[overlay] cold-pass: {args.cold_prefix}_*.csv (gpu {args.gpu}, "
        f"iter {args.cold_iter})\n")
    mapping, phase_kernel_lists = load_cold_pass(
        args.cold_prefix, args.gpu, args.cold_iter)

    if args.dump_map:
        with open(args.dump_map, "w") as f:
            json.dump(mapping, f, indent=2)
        sys.stderr.write(f"  dumped mapping to {args.dump_map}\n")

    sys.stderr.write(f"[overlay] applying to {args.native_json}\n")
    no_explode = args.no_explode_phases
    overlay(args.native_json, mapping, out_path,
            phase_kernel_lists=phase_kernel_lists,
            explode_graph=not (no_explode or args.no_explode_graph),
            explode_prefetch=not (no_explode or args.no_explode_prefetch),
            explode_defaultdp=not (no_explode or args.no_explode_defaultdp),
            explode_defaultmp=not (no_explode or args.no_explode_defaultmp))


if __name__ == "__main__":
    main()
