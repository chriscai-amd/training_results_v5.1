#!/usr/bin/env python3
"""HugeCTR train.py with a ProfilerCallback that brackets a specific iter
window via cudaProfilerStart/Stop. Use with:

    nsys profile -t cuda,nvtx,osrt,cudnn,cublas \\
        --capture-range=cudaProfilerApi \\
        --capture-range-end=stop \\
        ...

Set NSYS_TARGET_ITER and NSYS_NUM_ITERS env vars to control the window.
Defaults to iter 50 for 5 iters.

This is a thin wrapper around train.py — we re-import all the model setup
from there and only swap the callback list.
"""
from __future__ import annotations

import ctypes
import os
import threading
import time

# cudaProfilerStart / cudaProfilerStop bindings (must load BEFORE importing hugectr
# so that the same libcudart is used).
_cudart = ctypes.CDLL("libcudart.so")
_cudart.cudaProfilerStart.restype = ctypes.c_int
_cudart.cudaProfilerStop.restype = ctypes.c_int


import hugectr  # noqa: E402


class ProfilerWindowCallback(hugectr.TrainingCallback):
    """Spawns a background thread on training_start that calls cudaProfilerStart
    after ``delay_s`` seconds and cudaProfilerStop after another ``duration_s``
    seconds. The exact iter window depends on ``est_iter_ms`` (per-iter
    measurement from a prior unprofiled run).
    """

    def __init__(self, target_iter: int, num_iters: int, est_iter_ms: float):
        self.target_iter = target_iter
        self.num_iters = num_iters
        self.delay_s = target_iter * est_iter_ms / 1000.0
        self.duration_s = num_iters * est_iter_ms / 1000.0
        self._stopped = False
        super().__init__()

    def _driver(self):
        time.sleep(self.delay_s)
        rc1 = _cudart.cudaProfilerStart()
        t0 = time.perf_counter()
        time.sleep(self.duration_s)
        rc2 = _cudart.cudaProfilerStop()
        t1 = time.perf_counter()
        self._stopped = True
        # Note: in --capture-range=cudaProfilerApi --capture-range-end=stop mode,
        # nsys terminates when cudaProfilerStop fires. So this print may not flush.
        print(
            f"[ProfilerWindowCallback] window: target_iter={self.target_iter}, "
            f"duration={t1 - t0:.3f}s, rc_start={rc1}, rc_stop={rc2}",
            flush=True,
        )

    def on_training_start(self):
        print(
            f"[ProfilerWindowCallback] sleeping {self.delay_s:.3f}s before "
            f"cudaProfilerStart (target iter {self.target_iter}, "
            f"recording {self.num_iters} iters)",
            flush=True,
        )
        threading.Thread(target=self._driver, daemon=True).start()

    def on_training_end(self, current_iter: int):
        if not self._stopped:
            _cudart.cudaProfilerStop()


# Now run train.py via exec, after monkey-patching the callback list. We can't
# import train.py because it executes argparse / model setup at import time.
# Easiest: just exec it and inject our callback into the trainings_callbacks
# list before model construction.
_target_iter = int(os.environ.get("NSYS_TARGET_ITER", "50"))
_num_iters = int(os.environ.get("NSYS_NUM_ITERS", "5"))
_est_iter_ms = float(os.environ.get("NSYS_EST_ITER_MS", "20"))

_profiler_cb = ProfilerWindowCallback(_target_iter, _num_iters, _est_iter_ms)

# Monkey-patch hugectr.CreateSolver to inject our callback alongside the existing
# training_callbacks list passed by train.py.
_original_create_solver = hugectr.CreateSolver


def _patched_create_solver(*args, **kwargs):
    cbs = list(kwargs.get("training_callbacks", []) or [])
    cbs.append(_profiler_cb)
    kwargs["training_callbacks"] = cbs
    print(
        f"[ProfilerWindowCallback] injecting into solver alongside "
        f"{len(cbs) - 1} existing callbacks",
        flush=True,
    )
    return _original_create_solver(*args, **kwargs)


hugectr.CreateSolver = _patched_create_solver

# Now exec the original train.py with the patched solver.
_train_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train.py")
with open(_train_path) as f:
    _train_src = f.read()

exec(compile(_train_src, _train_path, "exec"), {"__name__": "__main__"})
