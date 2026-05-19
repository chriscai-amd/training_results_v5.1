#!/usr/bin/env python3
"""Merge per-rank HCTR native Perfetto JSONs into one 8-GPU trace.

Phase 20.9e: rebase timestamps to start at 0 so Perfetto's auto-fit zoom
doesn't include leading empty time from process startup.
"""
import argparse
import glob
import json
import os
import sys


def merge(src_dir: str, out_name: str = "hctr_native_trace_all8gpus.json") -> str:
    files = sorted(glob.glob(os.path.join(src_dir, "hctr_native_trace_rank*.json")))
    if not files:
        sys.exit(f"no rank files found under {src_dir}")
    all_evs = []
    for f in files:
        all_evs.extend(json.load(open(f))["traceEvents"])

    # Phase 20.9e: rebase timestamps so the trace starts at ts=0. Removes
    # the long leading empty span caused by absolute std::chrono timestamps
    # (steady_clock counts from system boot -> huge numbers).
    real_ts = [e["ts"] for e in all_evs if e.get("ph") == "X" and "ts" in e]
    if real_ts:
        base_ts = min(real_ts)
        for e in all_evs:
            if "ts" in e and isinstance(e["ts"], (int, float)) and e["ts"] > 0:
                e["ts"] = e["ts"] - base_ts

    # Phase 20.9f: Perfetto auto-hashes string tids into numeric ids and the
    # `thread_sort_index` metadata doesn't always stick. Translate string tids
    # into small fixed numeric tids based on the desired display order, then
    # add `thread_name` metadata to keep the human-readable lane label.
    # Numeric tids sort naturally low->high in Perfetto's lane list.
    tid_order = {
        "iter":       0,  # top
        "host_api":   1,
        "default":    2,  # main compute
        "defaultmp":  3,
        "defaultdp":  4,
        "comm":       5,
        "prefetch":   9,  # bottom
    }
    DEFAULT_TID = 50
    for e in all_evs:
        t = e.get("tid")
        if isinstance(t, str):
            e["tid"] = tid_order.get(t, DEFAULT_TID + (hash(t) % 50))
            # stash original name for the metadata pass below
            e.setdefault("_orig_tid", t)

    ranks = sorted(set(e.get("pid", 0) for e in all_evs))
    for r in ranks:
        all_evs.append({"name": "process_name", "ph": "M", "pid": r, "tid": 0,
                         "args": {"name": f"Rank {r} (GPU {r})"}})
        all_evs.append({"name": "process_sort_index", "ph": "M", "pid": r, "tid": 0,
                         "args": {"sort_index": r}})
        # Per-thread name metadata. Collect (numeric_tid -> original name).
        rank_tid_names = {}
        for e in all_evs:
            if e.get("pid") == r and "_orig_tid" in e:
                rank_tid_names.setdefault(e["tid"], e["_orig_tid"])
        for ntid, name in sorted(rank_tid_names.items()):
            all_evs.append({"name": "thread_name", "ph": "M",
                            "pid": r, "tid": ntid,
                            "args": {"name": name}})
            all_evs.append({"name": "thread_sort_index", "ph": "M",
                            "pid": r, "tid": ntid,
                            "args": {"sort_index": ntid}})

    # Strip the bookkeeping field before writing.
    for e in all_evs:
        e.pop("_orig_tid", None)

    # Sort by ts for nicer rendering.
    all_evs.sort(key=lambda e: (e.get("ts", 0), e.get("pid", 0)))

    out_path = os.path.join(src_dir, out_name)
    with open(out_path, "w") as f:
        json.dump({"displayTimeUnit": "ms", "traceEvents": all_evs}, f,
                    separators=(",", ":"))
    print(f"merged: {out_path}  ({os.path.getsize(out_path) // 1024} KB, "
          f"{len(all_evs)} events, span "
          f"{max(e.get('ts', 0) + e.get('dur', 0) for e in all_evs) / 1000:.0f} ms)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("src_dir", help="dir containing hctr_native_trace_rank*.json")
    p.add_argument("--out", default="hctr_native_trace_all8gpus.json")
    a = p.parse_args()
    merge(a.src_dir, a.out)
