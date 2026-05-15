#include "hip/hip_runtime.h"
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <data_readers/multi_hot/detail/system_latch.hpp>

namespace HugeCTR {

template <typename T>
__global__ void kernel_count_down(T* latch, T n) {
  // HIP/ROCm: atomicDec_system (system-scope atomic) is not exposed as a
  // standalone intrinsic; use the C11 atomic builtin directly with system scope.
  // This compiles and runs on AMD; semantics match CUDA atomicDec_system as long
  // as the latch is in managed/host-pinned memory (which SystemLatch ensures
  // via hipMallocManaged).
  __atomic_fetch_sub(latch, n, __ATOMIC_SEQ_CST);
}

void SystemLatch::device_count_down(hipStream_t stream, value_type n, bool from_graph) {
  if (from_graph) {
    kernel_count_down<<<1, 1, 0, stream>>>(latch_, n);
  } else {
    hipStreamAddCallback(stream, &SystemLatch::callback, (void*)latch_, 0);
  }
}

}  // namespace HugeCTR