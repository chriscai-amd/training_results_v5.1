#!/usr/bin/env bash
set -eu
apt-get update -qq 2>&1 | tail -1
# libaio-dev preferred; libaio1t64 is the noble fallback. Same for tbb.
# Bury exit codes since the second variant may not be needed.
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
  || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
  || true
# Install pybind11 + numpy via SYSTEM python3 (matches /usr/bin/python3 used by cmake)
/usr/bin/python3 -m pip install --break-system-packages --quiet pybind11 numpy 2>&1 | tail -1 || true
echo "[ok] deps installed"
mkdir -p /workspace/hugectr_hip/build_rocm72
cd /workspace/hugectr_hip/build_rocm72
export PATH=/opt/rocm/bin:$PATH
# Pin to /usr/bin/python3 (system python with pre-installed python3-dev headers
# at /usr/include/python3.12/). The docker image also has /opt/venv/bin/python3
# which is a symlink to /usr/bin/python3.12 BUT reports sys.prefix=/opt/venv,
# causing cmake's find_package(Python) to look for headers under
# /opt/venv/include/python3.12/ which does NOT exist. `which python3` resolves
# to the venv path and produces broken libs (segfault at AUC NCCL warm-up).
PYBIND11_DIR=$(/usr/bin/python3 -c "import pybind11; print(pybind11.get_cmake_dir())" 2>/dev/null || true)
# Match the proven-working scripts/build_hctr_rocm712.sh invocation. Default
# `cmake ..` without these flags produces libs that segfault / throw
# "Runtime error: invalid argument" at AUC NCCL warm-up because -O3 + DNDEBUG
# + gfx950 offload + MULTINODES=OFF + explicit Python path are all required
# for ABI-correct + assert-free output.
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=/opt/rocm/lib/llvm/bin/clang \
      -DCMAKE_CXX_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_COMPILER=/opt/rocm/lib/llvm/bin/clang++ \
      -DCMAKE_HIP_ARCHITECTURES=gfx950 \
      -DCMAKE_PREFIX_PATH=/opt/rocm \
      -DCMAKE_HIP_FLAGS="-O3 -DNDEBUG --offload-arch=gfx950" \
      -DENABLE_MULTINODES=OFF \
      -DSM=950 \
      -DPython_EXECUTABLE=/usr/bin/python3 \
      ${PYBIND11_DIR:+-Dpybind11_DIR=$PYBIND11_DIR} \
      .. > /tmp/cm.log 2>&1
grep -E 'HugeCTR ROCm port: linking against' /tmp/cm.log
echo ""
echo "=== build ==="
cmake --build . -j 16 2>&1 | tail -15
echo ""
ls -lh lib/*.so 2>/dev/null
