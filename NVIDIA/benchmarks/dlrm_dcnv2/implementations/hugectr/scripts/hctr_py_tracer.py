"""
Minimal, low-overhead Python frame tracer that produces a Chrome-trace
sidecar JSON consumable by nsys_to_perfetto_annotated.py via --py-trace.

Activation:
    HCTR_PY_TRACE=1                  -> enable; output goes to
                                        $HCTR_PY_TRACE_OUT
                                        (default /tmp/py_trace_rank{rank}.json)
    HCTR_PY_TRACE_OUT=/path/to.json  -> override output path
    HCTR_PY_TRACE_FILTER=path1:path2 -> ":"-separated substrings; ONLY frames
                                        whose code filename contains any of
                                        these are emitted (default:
                                        "train_mi350.py:hugectr:dlrm_dcnv2"
                                        to skip stdlib/numpy noise)
    HCTR_PY_TRACE_DEPTH=N            -> stop emitting deeper than N (0=root,
                                        default 30)

Time base:
    Each frame's `ts_realtime_ns` is wall-clock ns (CLOCK_REALTIME, i.e.
    "ns since UTC epoch"). This is the SAME clock that nsys uses for its
    TARGET_INFO_SESSION_START_TIME.utcEpochNs row, which lets the
    converter rewrite ts into nsys-window-relative microseconds without
    any extra alignment markers. As long as the Python tracer runs
    inside the same nsys session, alignment is exact.

Output schema (consumed by nsys_to_perfetto_annotated.py):
    {
      "clock": "CLOCK_REALTIME",
      "anchor_realtime_ns": <int>,
      "rank": <int>,
      "traceEvents": [
        {"name": "<file>:<line> <func>",
         "ph": "X",
         "ts_realtime_ns": <int>,
         "dur_ns": <int>,
         "rank": <int>,
         "args": {"file": "...", "line": ..., "func": "..."}},
        ...
      ]
    }

Usage from train_mi350.py:
    import os
    if os.environ.get("HCTR_PY_TRACE") == "1":
        from scripts.hctr_py_tracer import start, dump
        start()
        try:
            run_training(...)
        finally:
            dump()

The tracer uses sys.setprofile (per-call C-level hook, ~250 ns/call
overhead). For a 30-second training warm-up this adds at most a few %
to Python-side time, and zero to GPU steady-state time (Python is
quiescent during graph replay).
"""
from __future__ import annotations
import json
import os
import sys
import threading
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

_state = {
    "enabled": False,
    "rank": 0,
    "out_path": None,
    "filter_substrings": (),
    "max_depth": 30,
    "events": [],          # list of (ts_realtime_ns, ph, name, args)
    "anchor_realtime_ns": 0,
    "stack_per_thread": defaultdict(list),  # tid -> list of (ts, name, args)
}


def _now_realtime_ns() -> int:
    return time.clock_gettime_ns(time.CLOCK_REALTIME)


def _frame_label(frame) -> tuple[str, dict]:
    code = frame.f_code
    fn = code.co_filename
    ln = frame.f_lineno
    func = code.co_name
    short_fn = fn.rsplit("/", 2)
    short = "/".join(short_fn[-2:]) if len(short_fn) > 1 else fn
    name = f"{short}:{ln} {func}"
    args = {"file": fn, "line": ln, "func": func}
    return name, args


def _profile_hook(frame, event, arg):
    st = _state
    if not st["enabled"]:
        return
    if event not in ("call", "return", "c_call", "c_return"):
        return
    code = frame.f_code
    fn = code.co_filename
    if st["filter_substrings"]:
        if not any(s in fn for s in st["filter_substrings"]):
            return
    tid = threading.get_ident()
    stack = st["stack_per_thread"][tid]
    if len(stack) >= st["max_depth"] and event in ("call", "c_call"):
        return
    if event in ("call", "c_call"):
        name, args = _frame_label(frame)
        if event == "c_call":
            name = f"<c_call> {arg.__name__ if hasattr(arg, '__name__') else 'cfunc'}"
            args = {"cfunc": str(arg)}
        stack.append((_now_realtime_ns(), name, args, tid))
    elif event in ("return", "c_return") and stack:
        ts_start, name, args, tid_open = stack.pop()
        ts_end = _now_realtime_ns()
        st["events"].append({
            "name": name,
            "ph": "X",
            "ts_realtime_ns": ts_start,
            "dur_ns": ts_end - ts_start,
            "rank": st["rank"],
            "tid_native": tid_open,
            "args": args,
        })


def start(out_path: Optional[str] = None,
          rank: Optional[int] = None,
          filter_substrings: Optional[List[str]] = None,
          max_depth: Optional[int] = None) -> None:
    """Begin tracing. Idempotent; second call replaces config."""
    st = _state
    if rank is None:
        rank = int(os.environ.get("OMPI_COMM_WORLD_RANK")
                   or os.environ.get("SLURM_PROCID")
                   or os.environ.get("RANK")
                   or 0)
    if out_path is None:
        out_path = os.environ.get(
            "HCTR_PY_TRACE_OUT",
            f"/tmp/py_trace_rank{rank}.json")
    if filter_substrings is None:
        f = os.environ.get(
            "HCTR_PY_TRACE_FILTER",
            "train_mi350.py:hugectr:dlrm_dcnv2:multi_hot.py")
        filter_substrings = tuple(s for s in f.split(":") if s)
    if max_depth is None:
        max_depth = int(os.environ.get("HCTR_PY_TRACE_DEPTH", "30"))
    st["rank"] = rank
    st["out_path"] = out_path
    st["filter_substrings"] = tuple(filter_substrings)
    st["max_depth"] = max_depth
    st["events"] = []
    st["anchor_realtime_ns"] = _now_realtime_ns()
    st["enabled"] = True
    sys.setprofile(_profile_hook)
    threading.setprofile(_profile_hook)
    print(f"[hctr_py_tracer] started: rank={rank} "
          f"out={out_path} filter={filter_substrings}",
          file=sys.stderr)


def stop() -> None:
    _state["enabled"] = False
    sys.setprofile(None)
    threading.setprofile(None)


def dump(out_path: Optional[str] = None) -> str:
    """Stop and write sidecar JSON. Returns the output path."""
    stop()
    st = _state
    if out_path is None:
        out_path = st["out_path"]
    payload = {
        "clock": "CLOCK_REALTIME",
        "anchor_realtime_ns": st["anchor_realtime_ns"],
        "rank": st["rank"],
        "n_events": len(st["events"]),
        "filter_substrings": list(st["filter_substrings"]),
        "traceEvents": st["events"],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f)
    print(f"[hctr_py_tracer] wrote {len(st['events']):,} events -> {out_path}",
          file=sys.stderr)
    return out_path


def maybe_autostart() -> bool:
    """Helper for opt-in start via env var.
    Returns True if tracing was enabled, False otherwise."""
    if os.environ.get("HCTR_PY_TRACE") == "1":
        start()
        import atexit
        atexit.register(dump)
        return True
    return False
