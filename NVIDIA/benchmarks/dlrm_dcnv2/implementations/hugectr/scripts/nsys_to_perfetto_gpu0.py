#!/usr/bin/env python3
"""
Convert one GPU's events from an nsys SQLite export to a Chrome / Perfetto
JSON trace.

Output is Chrome Trace Event Format ("traceEvents" array). Perfetto UI
(https://ui.perfetto.dev) accepts this directly via drag-drop.

Usage:
    python3 nsys_to_perfetto_gpu0.py <input.sqlite> <gpu_device_id> [output.json]

The script keeps:
  - All CUDA kernels on the requested deviceId (separated per stream as a
    different "tid"; appears as a separate lane in Perfetto).
  - All cuda{Memcpy,Memset}Async events on that device.
  - cudaGraphLaunch / cudaStreamSynchronize / cudaLaunchKernel host-side
    events (mapped to a "host" process in Perfetto).
  - String IDs are looked up so kernel names appear directly.
"""
import sqlite3
import sys
import json
import os

if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)

DB = sys.argv[1]
GPU = int(sys.argv[2])
OUT = sys.argv[3] if len(sys.argv) > 3 else f"{os.path.splitext(DB)[0]}.gpu{GPU}.perfetto.json"

con = sqlite3.connect(DB)
cur = con.cursor()

# String table
cur.execute("SELECT id, value FROM StringIds")
str_map = {sid: val for sid, val in cur.fetchall()}

# Earliest start across all kernels (so we re-zero the time axis)
cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE deviceId=?", (GPU,))
t_min_kernel = cur.fetchone()[0] or 0
cur.execute("SELECT MIN(start) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE deviceId=?", (GPU,))
t_min_memcpy = cur.fetchone()[0] or t_min_kernel
T0 = min(t_min_kernel, t_min_memcpy)
print(f"T0 (ns) = {T0}", file=sys.stderr)

events = []
PID_GPU = 1   # one "process" per GPU
PID_HOST = 0

# --- Kernels on this GPU, lanes = streams ---
print(f"Loading kernels on GPU {GPU} ...", file=sys.stderr)
cur.execute("""
    SELECT start, end, streamId, shortName, demangledName
    FROM CUPTI_ACTIVITY_KIND_KERNEL
    WHERE deviceId = ?
    ORDER BY start
""", (GPU,))
n_kernels = 0
for start, end, sid, short_id, demangled_id in cur.fetchall():
    name = str_map.get(short_id) or str_map.get(demangled_id) or f"kernel_{short_id}"
    # Trim long mangled names
    if len(name) > 96:
        name = name[:93] + "..."
    events.append({
        "name": name,
        "cat": "kernel",
        "ph": "X",
        "ts": (start - T0) / 1000.0,    # ns -> us
        "dur": (end - start) / 1000.0,
        "pid": PID_GPU,
        "tid": sid,
    })
    n_kernels += 1
print(f"  {n_kernels:,} kernels", file=sys.stderr)

# --- Memcpy events on this GPU ---
print(f"Loading memcpys on GPU {GPU} ...", file=sys.stderr)
cur.execute("""
    SELECT start, end, streamId, copyKind
    FROM CUPTI_ACTIVITY_KIND_MEMCPY
    WHERE deviceId = ?
    ORDER BY start
""", (GPU,))
COPY_KINDS = {1: "memcpy_HtoD", 2: "memcpy_DtoH", 3: "memcpy_DtoD",
              8: "memcpy_HtoD_async", 9: "memcpy_DtoH_async",
              10: "memcpy_DtoD_async"}
n_memcpy = 0
for start, end, sid, kind in cur.fetchall():
    events.append({
        "name": COPY_KINDS.get(kind, f"memcpy_kind{kind}"),
        "cat": "memcpy",
        "ph": "X",
        "ts": (start - T0) / 1000.0,
        "dur": (end - start) / 1000.0,
        "pid": PID_GPU,
        "tid": sid,
    })
    n_memcpy += 1
print(f"  {n_memcpy:,} memcpys", file=sys.stderr)

# --- Host-side CUDA API calls (one rank only - rank that owns this GPU) ---
print(f"Loading host-side CUDA API calls (filtered to API names of interest)...", file=sys.stderr)
WANTED_API = ["cudaGraphLaunch", "cudaLaunchKernel", "cudaStreamSynchronize",
              "cudaMemcpyAsync", "cudaEventSynchronize", "cudaDeviceSynchronize"]
wanted_ids = [sid for sid, val in str_map.items()
              if any(val.startswith(w) for w in WANTED_API)]
ph = ",".join("?" * len(wanted_ids))

# A guess: pick the host-thread (globalTid) that issues most cudaGraphLaunches
# on this GPU. We don't have a perfect "which-rank-owns-which-GPU" mapping
# without reading TARGET_INFO_CUDA_CONTEXT_INFO. Heuristic: use the first
# globalTid we see.
gid_for_graphlaunch = [sid for sid, val in str_map.items()
                       if val == "cudaGraphLaunch_v10000"]
if gid_for_graphlaunch:
    cur.execute(f"""
        SELECT globalTid, COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE nameId = ? GROUP BY globalTid ORDER BY 2 DESC LIMIT 8
    """, (gid_for_graphlaunch[0],))
    tids = [r[0] for r in cur.fetchall()]
    print(f"  globalTids issuing cudaGraphLaunch (top 8): {tids}", file=sys.stderr)
    pick_tid = tids[GPU] if GPU < len(tids) else tids[0]
    print(f"  picking globalTid={pick_tid} for GPU {GPU}", file=sys.stderr)

    cur.execute(f"""
        SELECT start, end, nameId
        FROM CUPTI_ACTIVITY_KIND_RUNTIME
        WHERE globalTid = ? AND nameId IN ({ph})
        ORDER BY start
    """, [pick_tid] + wanted_ids)
    n_api = 0
    for start, end, nid in cur.fetchall():
        events.append({
            "name": str_map.get(nid, f"api_{nid}"),
            "cat": "host_api",
            "ph": "X",
            "ts": (start - T0) / 1000.0,
            "dur": (end - start) / 1000.0,
            "pid": PID_HOST,
            "tid": 1,
        })
        n_api += 1
    print(f"  {n_api:,} host API events", file=sys.stderr)

# --- Process / thread name metadata (Perfetto displays nicely) ---
events.append({"name": "process_name", "ph": "M", "pid": PID_GPU, "tid": 0,
               "args": {"name": f"GPU {GPU}"}})
events.append({"name": "process_name", "ph": "M", "pid": PID_HOST, "tid": 0,
               "args": {"name": f"Host (rank {GPU})"}})

# stream_id -> "stream <id>" thread label
unique_streams = sorted(set(e["tid"] for e in events if e["pid"] == PID_GPU))
for sid in unique_streams:
    events.append({"name": "thread_name", "ph": "M", "pid": PID_GPU, "tid": sid,
                   "args": {"name": f"stream {sid}"}})

print(f"\nTotal events: {len(events):,}", file=sys.stderr)
print(f"Writing {OUT} ...", file=sys.stderr)
with open(OUT, "w") as f:
    json.dump({"traceEvents": events,
               "displayTimeUnit": "ms",
               "metadata": {"src": DB, "gpu": GPU, "T0_ns": T0}}, f)
sz = os.path.getsize(OUT) / 1e6
print(f"Wrote {OUT}  ({sz:.1f} MB)", file=sys.stderr)

con.close()
