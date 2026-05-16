#!/usr/bin/env bash
# Inner script: incremental build of HCTR with the Phase-19 patch.
# Runs inside the container; build dir is /workspace/hugectr_hip/build_rocm72
# (shared NFS so the resulting .so are visible from any other node).
set -eu
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
/usr/bin/python3 -m pip install --break-system-packages --quiet pybind11 numpy 2>&1 | tail -1 || true
echo "===== Phase-19 incremental build ====="
cd /workspace/hugectr_hip/build_rocm72
export PATH=/opt/rocm/bin:$PATH
T0=$(date +%s)
cmake --build . -j 16 2>&1 | tail -10
T1=$(date +%s)
echo "[ok] build took $((T1-T0))s"
ls -lh lib/hugectr.so lib/libembedding.so 2>/dev/null
echo "[ok] build complete"
