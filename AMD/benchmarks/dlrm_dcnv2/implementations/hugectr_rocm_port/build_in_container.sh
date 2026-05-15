#!/usr/bin/env bash
set -eu
apt-get update -qq 2>&1 | tail -1
apt-get install -y -qq libaio-dev libnuma-dev libtbb-dev 2>&1 | tail -1
echo "[ok] deps installed"
cd /workspace/hugectr_hip/build_rocm72
export PATH=/opt/rocm/bin:$PATH
cmake .. > /tmp/cm.log 2>&1
grep -E 'HugeCTR ROCm port: linking against' /tmp/cm.log
echo ""
echo "=== build ==="
cmake --build . -j 16 2>&1 | tail -15
echo ""
ls -lh lib/*.so 2>/dev/null
