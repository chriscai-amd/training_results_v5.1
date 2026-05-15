#!/usr/bin/env bash
# Run the Criteo training smoke inside the AMD nightly container, against the
# freshly-built ROCm 7.2 binaries.
set -eu
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 > /dev/null 2>&1 || true

export LD_LIBRARY_PATH=/opt/rocm/lib:/workspace/hugectr_hip/build_rocm72/lib
export PYTHONPATH=/workspace/hugectr_hip/build_rocm72/lib:${PYTHONPATH:-}

cd /workspace/runtime_test
apt-get install -y -qq gdb > /dev/null 2>&1 || true
ulimit -c 0
echo "=== python + hugectr versions ==="
python3 -c "
import sys
print('python', sys.version)
import hugectr
print('hugectr from', hugectr.__file__)
print('  Adam:', hugectr.Optimizer_t.Adam)
"

echo ""
echo "=== run training (EmbeddingCollection API) ==="
python3 python_criteo_train_ec.py
