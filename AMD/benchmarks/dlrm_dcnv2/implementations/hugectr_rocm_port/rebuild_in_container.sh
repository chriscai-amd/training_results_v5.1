#!/usr/bin/env bash
# Rebuild HugeCTR-on-ROCm inside the AMD nightly container (ROCm 7.2.1).
# This gets us:
#   - hipBLASLt 1.2 with 443 gfx950 TensileLibrary entries (vs 100 in 7.0.2)
#   - Python 3.12 + pybind11 + pre-installed PyTorch
#   - All ROCm libs in their newer versions
set -eu

# Install libaio if missing (needed by multi_hot async data reader).
# Note: the container is ephemeral, so this runs every rebuild. Use full
# output (no -qq) so failures are visible. Don't `set -e` around apt because
# `apt-get update` may emit non-fatal GPG warnings.
if [ ! -f /usr/include/libaio.h ]; then
    echo "[..] installing libaio-dev"
    set +e
    apt-get update 2>&1 | tail -5
    apt-get install -y libaio-dev 2>&1 | tail -10
    set -e
    if [ ! -f /usr/include/libaio.h ]; then
        echo "[FATAL] libaio-dev install failed; cannot build the multi-hot reader."
        exit 2
    fi
fi

cd /workspace/hugectr_hip

# Reuse our existing CMakeLists; just point Python to the container's interpreter.
mkdir -p build_rocm72
cd build_rocm72

export PATH=/opt/rocm/bin:$PATH
export HIP_PLATFORM=amd
export CC=/opt/rocm/bin/amdclang
export CXX=/opt/rocm/bin/amdclang++

# Update the Python_EXECUTABLE in the parent CMakeLists temporarily via cache override.
echo "=== cmake configure (ROCm 7.2.1) ==="
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_HIP_ARCHITECTURES=gfx950 \
      -DPython_EXECUTABLE=/opt/venv/bin/python3 \
      .. 2>&1 | tail -25

echo ""
echo "=== build all ==="
cmake --build . -j 32 2>&1 | tail -30
echo ""
ls -lh lib/*.so 2>/dev/null
