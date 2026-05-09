# Minimal stub of mlperf_common's HCTRCommunicationHandler + MLLoggerWrapper.
# The real package wraps mlperf_logging for HugeCTR/Slurm/MPI integration; we
# only need the API surface that NVIDIA's train.py uses (see imports there:
#   from mlperf_common.frameworks.hugectr import HCTRCommunicationHandler
#   from mlperf_common.logging import MLLoggerWrapper
# Both are constructed once and used to emit MLLOG-format events).

class HCTRCommunicationHandler:
    """Single-process placeholder for the real Slurm/MPI handler."""
    def __init__(self, *a, **kw):
        pass
    def barrier(self):
        pass
    def is_master(self):
        return True
    def global_rank(self):
        return 0
    def world_size(self):
        return 1
    def local_rank(self):
        return 0


class MLLoggerWrapper:
    """Minimal MLLog-format event emitter for stdout/stderr."""
    def __init__(self, communication_handler=None, value=None, **kwargs):
        # Match upstream attribute name: callbacks.py reads self.mllogger.comm_handler.
        self.comm_handler = communication_handler or HCTRCommunicationHandler()
        self.handler = self.comm_handler

    def _emit(self, key, value=None, metadata=None, level="INFO"):
        import time as _t, json as _j
        ts = _t.time() * 1000
        meta = _j.dumps(metadata or {}, sort_keys=True)
        print(f":::MLLOG {{\"namespace\": \"hugectr\", \"time_ms\": {ts:.0f}, "
              f"\"event_type\": \"{level.upper()}\", \"key\": \"{key}\", "
              f"\"value\": {_j.dumps(value)}, \"metadata\": {meta}}}", flush=True)

    def start(self, key, value=None, metadata=None):
        self._emit(key, value, metadata, "INTERVAL_START")

    def end(self, key, value=None, metadata=None):
        self._emit(key, value, metadata, "INTERVAL_END")

    def event(self, key, value=None, metadata=None):
        self._emit(key, value, metadata, "POINT_IN_TIME")

    def mlperf_submission_log(self, benchmark, num_nodes, org):
        self._emit("submission_org", org)
        self._emit("submission_benchmark", benchmark)
        self._emit("submission_division", "closed")
        self._emit("submission_status", "research")
        self._emit("submission_platform", f"{num_nodes}-node AMD MI350X")

    # Compound helpers used by the upstream `mlperf_logger.callbacks` module
    # (which we don't replace, just feed via this stub).
    def log_init_stop_run_start(self):
        self.end("init_stop")
        self.start("run_start")

    def log_run_stop(self, status="aborted", epoch_num=0.0):
        self.end("run_stop", metadata={"status": status, "epoch_num": epoch_num})

    def log_eval(self, key, value, epoch_num=0.0, **kw):
        self.event(key, value, metadata={"epoch_num": epoch_num, **kw})

    def __getattr__(self, name):
        # Catch-all so the upstream callback never tripped on a missing method.
        def _stub(*a, **kw):
            self._emit(name, kw.get("value"), kw.get("metadata"))
        return _stub
