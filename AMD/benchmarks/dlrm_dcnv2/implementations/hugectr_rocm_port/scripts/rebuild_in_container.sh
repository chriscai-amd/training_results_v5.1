#!/usr/bin/env bash
# Rebuild HugeCTR-on-ROCm inside the AMD nightly container (ROCm 7.2.1).
# This gets us:
#   - hipBLASLt 1.2 with 443 gfx950 TensileLibrary entries (vs 100 in 7.0.2)
#   - Python 3.12 + pybind11 + pre-installed PyTorch
#   - All ROCm libs in their newer versions
set -eu

# Always install libaio-dev (needed by multi_hot async data reader at
# both compile and link time). Ubuntu Noble renamed the dev package
# to libaio1t64 in some images, so try both names. Unconditional and
# idempotent -- apt-get is a no-op when the packages are already
# present, and the noble container's libaio1t64 sometimes ships
# headers but not the libaio.so symlink the linker expects.
echo "[..] ensuring libaio-dev / libaio1t64 is installed"
apt-get update -qq 2>&1 | tail -3
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>/dev/null \
    || apt-get install -y -qq libaio1t64 libnuma-dev libtbb12 2>&1 | tail -1 \
    || true
# Sanity: linker needs either libaio.so or libaio.so.1 in the search path.
ldconfig -p 2>/dev/null | grep -E "libaio" | head -3 || echo "(no libaio in ldconfig cache yet)"

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
