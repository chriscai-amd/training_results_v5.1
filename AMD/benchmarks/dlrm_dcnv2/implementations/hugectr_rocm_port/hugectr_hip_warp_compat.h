// Compatibility shim for HugeCTR's CUDA -> HIP/ROCm port.
// Provides overloads for warp-level intrinsics whose signatures differ between
// CUDA (32-bit warp masks) and HIP on AMD (64-bit wavefront masks).
//
// Included by /apps/chcai/hugectr_rocm_port/hugectr_hip/CMakeLists.txt via
// add_compile_options(-include .../hugectr_hip_warp_compat.h) for HIP sources.
#pragma once
#ifdef __HIPCC__

#include <hip/hip_runtime.h>

// __ffs/__ffsll are device intrinsics. CUDA's __ffs takes (int|unsigned int).
// On AMD, cooperative_groups thread_block_tile<N>.ballot() returns
// `unsigned long long` (64-bit) regardless of N, so existing
// `__ffs(warp_tile.ballot(...))` calls become ambiguous.
//
// Forward 64-bit and `long` variants to __ffsll. Note: assumes the source kernel
// only uses the lower 32 bits when WARP_SIZE==32 (typical CUDA pattern).
#ifndef HUGECTR_HIP_FFS_COMPAT
#define HUGECTR_HIP_FFS_COMPAT
__device__ __forceinline__ int __ffs(unsigned long long x) {
  return __ffsll(static_cast<unsigned long long>(x));
}
__device__ __forceinline__ int __ffs(long long x) {
  return __ffsll(static_cast<unsigned long long>(x));
}
__device__ __forceinline__ int __ffs(unsigned long x) {
  return __ffsll(static_cast<unsigned long long>(x));
}
__device__ __forceinline__ int __ffs(long x) {
  return __ffsll(static_cast<unsigned long long>(x));
}
#endif // HUGECTR_HIP_FFS_COMPAT

// __popc / __popcll: same idea -- HIP only provides 32-bit __popc.
#ifndef HUGECTR_HIP_POPC_COMPAT
#define HUGECTR_HIP_POPC_COMPAT
__device__ __forceinline__ int __popc(unsigned long long x) {
  return __popcll(x);
}
__device__ __forceinline__ int __popc(unsigned long x) {
  return __popcll(static_cast<unsigned long long>(x));
}
#endif // HUGECTR_HIP_POPC_COMPAT

#endif // __HIPCC__
