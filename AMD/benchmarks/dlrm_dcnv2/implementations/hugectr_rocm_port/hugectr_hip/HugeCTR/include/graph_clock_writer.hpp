/*
 * Phase 20.9 (2026-05-18): GPU clock-writer kernel for in-graph timing.
 *
 * ROCm 7.2.1 has a confirmed bug (ROCm/rocm-systems#2380): event-record
 * nodes inserted via hipGraphAddEventRecordNode return 0us elapsedTime
 * when queried after replay. This makes the canonical
 * "hipGraphAddEventRecordNode + hipEventElapsedTime" approach unusable
 * for per-kernel timing inside captured HIP graphs on this driver.
 *
 * Workaround: insert KERNEL nodes (which DO work reliably in graphs)
 * that each write the GPU clock to a unique slot in a device buffer.
 * On host harvest, we read the slot values and compute per-kernel
 * durations from clock-tick deltas. Conversion to nanoseconds uses
 * hipDeviceGetAttribute(hipDeviceAttributeClockRate).
 *
 * Per-replay overhead: ~1us per clock-writer kernel (cheap GPU work,
 * single thread writes one u64). For 100 kernels: ~100us / iter = ~2.5%
 * iter overhead. Matches the rocprof / NV CUPTI overhead range.
 *
 * Usage:
 *   1. Allocate device buffer for N+1 u64 slots (1 per kernel node + start).
 *   2. After hipStreamEndCapture(), walk graph, find kernel nodes.
 *   3. For each kernel K, insert hipGraphAddKernelNode(
 *        params={func: graph_clock_writer_kernel, args: &buf[i]},
 *        deps={K}) -- writes clock AFTER K completes.
 *   4. Insert one writer as a graph root (no deps) for the "iter_start"
 *      tick (buf[0]).
 *   5. After each hipGraphLaunch, read buf to host, compute
 *        duration_us[i] = (buf[i+1] - buf[i]) / clock_rate_khz * 1000
 *   6. Emit per-kernel events to the PerfettoEmitter buffer.
 */
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace HugeCTR {
namespace tracing {

// Single-thread kernel: writes the current GPU clock (in cycles) to
// out[slot]. Cheap: one global store + clock64() read. Designed to be
// added as a graph node via hipGraphAddKernelNode.
__global__ void graph_clock_writer_kernel(unsigned long long* out, int slot);

}  // namespace tracing
}  // namespace HugeCTR
