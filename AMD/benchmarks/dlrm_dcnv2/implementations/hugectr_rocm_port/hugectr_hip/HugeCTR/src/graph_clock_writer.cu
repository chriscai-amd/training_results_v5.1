/*
 * Phase 20.9 (2026-05-18): see graph_clock_writer.hpp.
 */
#include "hip/hip_runtime.h"

#include <graph_clock_writer.hpp>

namespace HugeCTR {
namespace tracing {

__global__ void graph_clock_writer_kernel(unsigned long long* out, int slot) {
  // Single warp lead writes the GPU clock at this point in the stream.
  // wall_clock64() is a HIP intrinsic that returns the nanosecond-scale
  // wall clock from the GPU's shader unit. On AMD GFX, this maps to
  // s_memrealtime which is a single-cycle read of the global wall clock
  // (not the per-CU clock that clock() returns). This gives consistent
  // values across CUs, which we need for per-kernel duration measurement.
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    out[slot] = wall_clock64();
  }
}

}  // namespace tracing
}  // namespace HugeCTR
