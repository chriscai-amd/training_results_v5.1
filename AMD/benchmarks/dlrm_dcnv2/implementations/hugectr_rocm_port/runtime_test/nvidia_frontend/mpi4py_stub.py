# Minimal MPI-less shim for single-process runs.
# NVIDIA's train.py uses `from mpi4py import MPI; MPI.COMM_WORLD.{Get_size,Get_rank}`.
# Mock those for single-process execution.

class _Comm:
    def Get_size(self): return 1
    def Get_rank(self): return 0
    def barrier(self): pass


class _MPI:
    COMM_WORLD = _Comm()


MPI = _MPI()
