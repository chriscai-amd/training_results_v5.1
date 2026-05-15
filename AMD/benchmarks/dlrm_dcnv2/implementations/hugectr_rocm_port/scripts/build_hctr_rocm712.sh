#!/bin/bash
# Build HCTR against ROCm 7.12.0rc1 (preview). Run as root in a
# ghcr.io/rocm/no_rocm_image_ubuntu24_04 container with /apps/chcai
# and /home/chcai bind-mounted.
set -e

ROCM_DIR=/opt/rocm-7.12.0rc1
ln -sfn /apps/chcai/rocm_712_extracted ${ROCM_DIR}
ln -sfn ${ROCM_DIR} /opt/rocm
export PATH=${ROCM_DIR}/bin:${ROCM_DIR}/lib/llvm/bin:$PATH
export LD_LIBRARY_PATH=${ROCM_DIR}/lib

echo "=== install build deps ==="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq 2>&1 | tail -2 || true
# Skip the apt-key warnings — focus on what we need
apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build pkg-config \
    libnuma-dev libtbb-dev libaio-dev \
    libopenmpi-dev openmpi-bin \
    libstdc++-12-dev gcc g++ \
    python3-pip python3-dev python3-venv libpython3-dev \
    git curl ca-certificates \
    2>&1 | tail -5
echo "deps installed: $?"

echo
echo "=== quick header check (HIP segmented exec) ==="
grep -aE "Segmented|graph_segment" ${ROCM_DIR}/include/hip/hip_runtime_api.h 2>&1 | head -3 || echo "(no header strings; runtime-only)"

echo
echo "=== HCTR build dir setup (separate from /home/chcai/.../build_rocm72) ==="
WORKSPACE=/host/hugectr_rocm_port
BUILD_DIR=/apps/chcai/build_rocm712
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

echo
echo "=== cmake configure ==="
echo "=== make sure pybind11 + numpy installed in system python ==="
pip3 install --break-system-packages --quiet pybind11 numpy 2>&1 | tail -3 || pip3 install --quiet pybind11 numpy 2>&1 | tail -3

cmake \
    -DCMAKE_C_COMPILER=${ROCM_DIR}/lib/llvm/bin/clang \
    -DCMAKE_CXX_COMPILER=${ROCM_DIR}/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_COMPILER=${ROCM_DIR}/lib/llvm/bin/clang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx950 \
    -DCMAKE_PREFIX_PATH=${ROCM_DIR} \
    -DPython_EXECUTABLE=/usr/bin/python3 \
    -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_MULTINODES=OFF \
    -DSM=950 \
    -DCMAKE_HIP_FLAGS="-O3 -DNDEBUG --offload-arch=gfx950" \
    ${WORKSPACE}/hugectr_hip 2>&1 | tail -40
echo "configure exit=$?"
