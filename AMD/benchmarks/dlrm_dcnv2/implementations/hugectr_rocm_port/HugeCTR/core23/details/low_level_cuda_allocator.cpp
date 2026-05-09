/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 * Licensed under the Apache License, Version 2.0.
 *
 * HugeCTR -> ROCm port: this file originally implemented a CUDA driver-API
 * virtual-memory allocator with optional memory-compression support
 * (cuMemCreate/cuMemMap, CU_MEM_ALLOCATION_COMP_GENERIC). HIP exposes
 * hipMem* equivalents but lacks the compression-flag enum and has stricter
 * void* arithmetic rules. The simpler `SimpleCUDAAllocator` is the workhorse
 * used in practice; this file is stubbed to throw on any unexpected
 * allocation, which is detectable at runtime if anything *does* try to use it.
 */

#include <core23/details/low_level_cuda_allocator.hpp>
#include <core23/device.hpp>
#include <core23/error.hpp>
#include <core23/logger.hpp>
#include <core23/macros.hpp>

namespace HugeCTR {
namespace core23 {

LowLevelCUDAAllocator::LowLevelCUDAAllocator(const Device& device, bool /*compressible*/) {
  HCTR_THROW_IF(device.type() != DeviceType::GPU, HugeCTR::Error_t::IllegalCall,
                "Only DeviceType::GPU is supported.");
  HCTR_LOG_S(WARNING, ROOT)
      << "LowLevelCUDAAllocator (virtual-memory + compression) is stubbed in the "
      << "HugeCTR ROCm port. Use SimpleCUDAAllocator or PoolCUDAAllocator instead." << std::endl;
}

LowLevelCUDAAllocator::~LowLevelCUDAAllocator() {}

void* LowLevelCUDAAllocator::allocate(int64_t /*size*/, CUDAStream) {
  HCTR_THROW_IF(true, HugeCTR::Error_t::IllegalCall,
                "LowLevelCUDAAllocator::allocate is unsupported in the HugeCTR ROCm port. "
                "Use SimpleCUDAAllocator or PoolCUDAAllocator instead.");
  return nullptr;
}

void* LowLevelCUDAAllocator::do_resize(void* /*ptr*/, int64_t /*new_size*/, CUDAStream) {
  HCTR_THROW_IF(true, HugeCTR::Error_t::IllegalCall,
                "LowLevelCUDAAllocator::do_resize is unsupported in the HugeCTR ROCm port.");
  return nullptr;
}

void LowLevelCUDAAllocator::deallocate(void* /*ptr*/, CUDAStream) {
  // No-op: nothing was ever allocated.
}

int64_t LowLevelCUDAAllocator::default_alignment() const { return kcudaAllocationAlignment; }

}  // namespace core23
}  // namespace HugeCTR
