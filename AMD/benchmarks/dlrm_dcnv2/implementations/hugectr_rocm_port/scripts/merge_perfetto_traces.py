#!/usr/bin/env python3
"""Merge per-rank HCTR Perfetto JSONs into one 8-GPU trace.

Handles two input shapes:
  * raw native traces  (hctr_native_trace_rank*.json) — pid already set to rank R
  * overlay traces     (overlay_rank*.json)            — pid always 0, remap to R

Default behavior trims the first and last captured iter (--trim-edges 1).
The first traced iter often shows ~5x inflated duration due to hipEvent
staleness from untraced warmup iters (see docs/tracing.md "First traced
iter (BEGIN) distortion"). The last iter is frequently a partial-window
outlier. Both are dropped along with every event whose timestamp falls
inside their envelopes. Timestamps are then re-rebased so the first
kept iter starts at ts=0.

Phase 20.9e: rebase timestamps to start at 0 so Perfetto's auto-fit zoom
doesn't include leading empty time from process startup.
Phase 21.0  (2026-05-21): overlay-file support + edge-iter auto-trim.
"""
import argparse
import glob
import json
import os
import re
import sys


# Phase 20.9f: small fixed numeric tids so Perfetto sorts lanes the way we want.
TID_ORDER = {
    "iter":       0,  # top
    "host_api":   1,
    "default":    2,  # main compute
    "defaultmp":  3,
    "defaultdp":  4,
    "comm":       5,
    "prefetch":   9,  # bottom
}
DEFAULT_TID = 50

# Regexes for the two supported per-rank filename shapes.
NATIVE_RE = re.compile(r"hctr_native_trace_rank(\d+)\.json$")
OVERLAY_RE = re.compile(r"overlay_rank(\d+)\.json$")


def _detect_pattern(src_dir: str):
    """Return (file_list, pattern_name). Prefer overlay if present."""
    overlay = sorted(glob.glob(os.path.join(src_dir, "overlay_rank*.json")))
    if overlay:
        return overlay, "overlay"
    native = sorted(glob.glob(os.path.join(src_dir, "hctr_native_trace_rank*.json")))
    if native:
        return native, "native"
    sys.exit(f"no rank files found under {src_dir} "
             f"(looked for overlay_rank*.json and hctr_native_trace_rank*.json)")


def _rank_of(path: str):
    for r in (OVERLAY_RE, NATIVE_RE):
        m = r.search(os.path.basename(path))
        if m:
            return int(m.group(1))
    return None


def _iter_envelopes(all_evs, ref_pid):
    """Return sorted list of (name, ts, end_ts) for iter_N parent events on ref_pid."""
    envs = []
    for e in all_evs:
        if not isinstance(e, dict): continue
        if e.get("pid") != ref_pid: continue
        n = e.get("name", "")
        if n.startswith("iter_") and e.get("ph") == "X" and "ts" in e and "dur" in e:
            envs.append((n, e["ts"], e["ts"] + e["dur"]))
    envs.sort(key=lambda x: x[1])
    return envs


def _trim_edge_iters(all_evs, trim_first: int, trim_last: int, ref_pid):
    """Drop iter_X parent envelopes for the first `trim_first` and last
    `trim_last` iters, and every other event whose ts falls inside their
    windows. Returns (kept_events, dropped_iter_names)."""
    envs = _iter_envelopes(all_evs, ref_pid)
    if len(envs) < trim_first + trim_last + 1:
        # Not enough iters to safely trim (would leave 0). Bail.
        return all_evs, []
    trimmed = envs[:trim_first] + (envs[-trim_last:] if trim_last else [])
    drop_names = {n for n, _, _ in trimmed}
    drop_ranges = [(s, e) for _, s, e in trimmed]

    def in_drop(ts):
        for s, e in drop_ranges:
            if s <= ts <= e:
                return True
        return False

    kept = []
    for e in all_evs:
        if not isinstance(e, dict):
            kept.append(e)
            continue
        n = e.get("name", "")
        if n in drop_names:
            continue
        ts = e.get("ts")
        if isinstance(ts, (int, float)) and in_drop(ts):
            continue
        kept.append(e)
    return kept, drop_names


def _rebase_ts(all_evs):
    real = [e["ts"] for e in all_evs
            if isinstance(e, dict) and e.get("ph") == "X" and "ts" in e]
    if not real:
        return
    base = min(real)
    for e in all_evs:
        if isinstance(e, dict) and "ts" in e and isinstance(e["ts"], (int, float)) and e["ts"] > 0:
            e["ts"] = e["ts"] - base


def merge(src_dir: str,
          out_name: str = None,
          trim_first: int = 1,
          trim_last: int = 1) -> str:
    files, pattern = _detect_pattern(src_dir)
    print(f"[merge] pattern={pattern}  files={len(files)}")
    if out_name is None:
        out_name = ("overlay_all8gpus_trimmed.json" if pattern == "overlay"
                    else "hctr_native_trace_all8gpus.json")

    all_evs = []
    for f in files:
        rank = _rank_of(f)
        evs = json.load(open(f))["traceEvents"]
        # Overlay files all use pid=0; remap to the rank derived from filename
        # so all 8 ranks land in distinct Perfetto process lanes. Native files
        # already have pid=rank, but force-set anyway for consistency when the
        # filename rank is the source of truth.
        if rank is not None:
            for e in evs:
                if isinstance(e, dict):
                    e["pid"] = rank
        all_evs.extend(evs)
    print(f"[merge] total events before trim: {len(all_evs)}")

    # Phase 20.9e: first rebase (raw -> ts0) so the iter envelopes are at
    # human-readable offsets for the trim step.
    _rebase_ts(all_evs)

    ref_pid = 0  # rank 0 is reference for iter envelopes
    if trim_first or trim_last:
        envs_before = _iter_envelopes(all_evs, ref_pid)
        kept, dropped = _trim_edge_iters(all_evs, trim_first, trim_last, ref_pid)
        envs_kept = [e for e in envs_before if e[0] not in dropped]
        if dropped:
            print(f"[merge] trimmed iters (first={trim_first}, last={trim_last}):")
            for n, s, e in envs_before:
                tag = "DROP" if n in dropped else "keep"
                dur_ms = (e - s) / 1000
                print(f"        {tag}  {n:10s}  dur={dur_ms:.3f} ms")
            print(f"[merge] events after trim: {len(kept)}  (dropped {len(all_evs) - len(kept)})")
            all_evs = kept
            # Re-rebase so the first kept iter starts at ts=0
            _rebase_ts(all_evs)
        else:
            print(f"[merge] only {len(envs_before)} iters captured — too few to trim safely")

    # Phase 20.9f: numeric tids in display order.
    for e in all_evs:
        t = e.get("tid")
        if isinstance(t, str):
            e["tid"] = TID_ORDER.get(t, DEFAULT_TID + (hash(t) % 50))
            e.setdefault("_orig_tid", t)

    ranks = sorted(set(e.get("pid", 0) for e in all_evs if isinstance(e, dict)))
    for r in ranks:
        all_evs.append({"name": "process_name", "ph": "M", "pid": r, "tid": 0,
                        "args": {"name": f"Rank {r} (GPU {r})"}})
        all_evs.append({"name": "process_sort_index", "ph": "M", "pid": r, "tid": 0,
                        "args": {"sort_index": r}})
        rank_tid_names = {}
        for e in all_evs:
            if isinstance(e, dict) and e.get("pid") == r and "_orig_tid" in e:
                rank_tid_names.setdefault(e["tid"], e["_orig_tid"])
        for ntid, name in sorted(rank_tid_names.items()):
            all_evs.append({"name": "thread_name", "ph": "M",
                            "pid": r, "tid": ntid,
                            "args": {"name": name}})
            all_evs.append({"name": "thread_sort_index", "ph": "M",
                            "pid": r, "tid": ntid,
                            "args": {"sort_index": ntid}})

    for e in all_evs:
        if isinstance(e, dict):
            e.pop("_orig_tid", None)

    all_evs.sort(key=lambda e: (e.get("ts", 0) if isinstance(e, dict) else 0,
                                 e.get("pid", 0) if isinstance(e, dict) else 0))

    out_path = os.path.join(src_dir, out_name)
    with open(out_path, "w") as f:
        json.dump({"displayTimeUnit": "ms", "traceEvents": all_evs}, f,
                  separators=(",", ":"))
    span_ms = max((e.get("ts", 0) + e.get("dur", 0))
                  for e in all_evs if isinstance(e, dict)) / 1000
    print(f"[merge] wrote {out_path}  ({os.path.getsize(out_path)//1024} KB, "
          f"{len(all_evs)} events, span {span_ms:.0f} ms)")
    return out_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src_dir", help="dir containing overlay_rank*.json or "
                                   "hctr_native_trace_rank*.json")
    p.add_argument("--out", default=None,
                   help="output filename (in src_dir); default depends on pattern")
    p.add_argument("--trim-edges", type=int, default=1, metavar="N",
                   help="drop first N and last N iters along with events inside "
                        "their envelopes (default 1; use 0 to disable)")
    p.add_argument("--trim-first", type=int, default=None,
                   help="override trim-edges for the first-iter side only")
    p.add_argument("--trim-last", type=int, default=None,
                   help="override trim-edges for the last-iter side only")
    a = p.parse_args()
    tf = a.trim_first if a.trim_first is not None else a.trim_edges
    tl = a.trim_last  if a.trim_last  is not None else a.trim_edges
    merge(a.src_dir, a.out, trim_first=tf, trim_last=tl)
