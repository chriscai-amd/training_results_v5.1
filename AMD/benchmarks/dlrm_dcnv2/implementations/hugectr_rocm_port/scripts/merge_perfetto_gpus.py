#!/usr/bin/env python3
"""Merge per-GPU Perfetto JSONs into a single multi-GPU view.

Each input JSON was emitted by rocprofv3_to_perfetto_annotated.py with its
OWN T_START offset (so `ts` in that file is microseconds relative to that
GPU's first event). To merge, we re-base every GPU to a common T_START (the
global minimum across all GPUs), then assign each GPU a unique `pid` so
Perfetto shows them as N stacked process lanes labeled "GPU 0 .. GPU 7".

Usage:
    python3 merge_perfetto_gpus.py \\
        <rocprof_csv_prefix> \\
        <input_glob> \\
        <output.json> \\
        [first_iter] [n_iters]

Examples:
    python3 merge_perfetto_gpus.py \\
        /apps/chcai/trace_712_iter1000/cv350-rck-g03-c10-08.rck.dcgpu/1129 \\
        '/home/chcai/trace_712_iter1000_perfetto/iter_steady_gpu*.json' \\
        /home/chcai/trace_712_iter1000_perfetto/iter_steady_all8gpus.json \\
        95 5

The script needs `rocprofv3_to_perfetto_annotated.py` to be in the same
directory — it's invoked once per GPU to recover the per-file T_START
absolute timestamp (which the per-GPU JSONs don't store directly).
"""
import json, sys, glob, re, os
from collections import defaultdict

# Args
prefix     = sys.argv[1] if len(sys.argv) > 1 else \
             "/apps/chcai/trace_712_iter1000/cv350-rck-g03-c10-08.rck.dcgpu/1129"
input_glob = sys.argv[2] if len(sys.argv) > 2 else \
             "/home/chcai/trace_712_iter1000_perfetto/iter_steady_gpu*.json"
out_path   = sys.argv[3] if len(sys.argv) > 3 else \
             "/home/chcai/trace_712_iter1000_perfetto/iter_steady_all8gpus.json"
first_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 95
n_iters    = int(sys.argv[5]) if len(sys.argv) > 5 else 5

in_files = sorted(glob.glob(input_glob))
if not in_files:
    sys.exit(f"ERROR: no input files match: {input_glob}")

# Need the original T_START values from each file — they're in the script output,
# not in the JSON itself. Easier: re-run the script in a mode that prints them,
# OR cheat: use the X-event with min ts in each file and the agent_info CSV with
# absolute timestamps.
#
# But actually, each file has events with relative ts >= 0. The DELTA between
# files is what matters. We can recover the per-file T_START by re-loading the
# kernel CSV and finding earliest Start_Timestamp for that GPU, then aligning.
#
# Simpler approach: ALL files were produced from the SAME rocprof trace, and
# the per-file T_START values were recorded by the conversion script.
# Re-run the script BRIEFLY to grab those numbers.

import subprocess
script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "rocprofv3_to_perfetto_annotated.py")
if not os.path.exists(script):
    # fallback: assume installed alongside in scripts/
    script = "/home/chcai/training_results_v5.1/AMD/benchmarks/dlrm_dcnv2/" \
             "implementations/hugectr_rocm_port/scripts/" \
             "rocprofv3_to_perfetto_annotated.py"

n_gpus = len(in_files)
t_starts = {}  # gpu_id -> absolute ns
print(f"Recovering T_START for each of {n_gpus} GPUs…")
for gpu in range(n_gpus):
    # Run the script briefly to extract the T_START line
    out = subprocess.run(
        ["python3", script, prefix, str(gpu), str(first_iter), str(n_iters), "/tmp/_dummy.json"],
        capture_output=True, text=True, timeout=60
    )
    m = re.search(r"T_START=(\d+)ns", out.stdout + out.stderr)
    if m:
        t_starts[gpu] = int(m.group(1))
        print(f"  GPU {gpu}: T_START = {t_starts[gpu]} ns")
    else:
        print(f"  GPU {gpu}: T_START NOT FOUND")
        sys.exit(1)
os.remove("/tmp/_dummy.json")

global_t_start = min(t_starts.values())
print(f"\nGlobal T_START = {global_t_start} ns (= GPU {min(t_starts, key=t_starts.get)})")
print(f"Per-GPU offsets (us, applied to all events in each file):")
for gpu in range(n_gpus):
    offset_us = (t_starts[gpu] - global_t_start) / 1000.0
    print(f"  GPU {gpu}: +{offset_us:.3f} us")
print()

merged_events = []
metadata_pids_added = set()

for gpu, in_file in enumerate(in_files):
    with open(in_file) as f:
        data = json.load(f)
    offset_us = (t_starts[gpu] - global_t_start) / 1000.0
    new_pid = 100 + gpu  # 100, 101, …, 107 — make distinct from the input's pid=1

    n = 0
    for e in data["traceEvents"]:
        e2 = dict(e)
        # Shift timestamp by per-GPU offset
        if "ts" in e2:
            e2["ts"] = e2["ts"] + offset_us
        # Remap pid uniformly to per-GPU pid
        if "pid" in e2:
            e2["pid"] = new_pid
        merged_events.append(e2)
        n += 1

    # Add a process_name metadata event for this GPU
    merged_events.append({
        "ph": "M", "name": "process_name", "pid": new_pid, "tid": 0,
        "args": {"name": f"GPU {gpu} (Agent {gpu+2})"}
    })
    merged_events.append({
        "ph": "M", "name": "process_sort_index", "pid": new_pid, "tid": 0,
        "args": {"sort_index": gpu}
    })
    print(f"GPU {gpu}: merged {n} events at offset +{offset_us:.3f} us → pid={new_pid}")

print(f"\nTotal merged events: {len(merged_events)}")
out = {"traceEvents": merged_events, "displayTimeUnit": "ns"}
with open(out_path, "w") as f:
    json.dump(out, f)
sz = os.path.getsize(out_path) / 1e6
print(f"\nWrote {out_path} ({sz:.2f} MB)")
