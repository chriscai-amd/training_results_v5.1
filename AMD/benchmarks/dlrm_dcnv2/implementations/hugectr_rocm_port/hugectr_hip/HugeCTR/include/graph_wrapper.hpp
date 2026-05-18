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
#pragma once

#include <hip/hip_runtime.h>

#include <functional>
#include <string>
#include <vector>

namespace HugeCTR {

struct GraphWrapper {
  bool initialized = false;
  hipGraph_t graph;
  hipGraphExec_t graph_exec;

  // Phase 20.8 (2026-05-18): per-kernel event-record nodes inserted
  // post-capture, before instantiation. Each entry is the event
  // recorded AFTER the corresponding kernel node in the captured
  // graph. Per-kernel duration = elapsedTime(events_[i-1], events_[i])
  // (or elapsedTime(graph_start_event_, events_[0]) for the first).
  //
  // BLOCKED ON ROCM 7.2.1: hipEventElapsedTime returns 0us for events
  // inserted via hipGraphAddEventRecordNode (ROCm/rocm-systems#2380).
  // Code stays env-gated OFF -- auto-tests when next ROCm lands.
  std::vector<hipEvent_t> per_kernel_events_;
  std::vector<std::string> per_kernel_names_;
  hipEvent_t graph_start_event_ = nullptr;
  int per_kernel_rank_ = -1;  // local rank for emit attribution

  // Phase 20.9 (2026-05-18): Workaround for the ROCm event-in-graph bug
  // using KERNEL nodes (which work reliably in graphs) that each write
  // the GPU wall_clock64() to a device buffer slot. On host harvest,
  // we read the buffer and compute per-kernel durations from clock
  // tick deltas.
  //
  // Per-replay overhead: ~1us per writer kernel (single-thread store).
  // Gated by HCTR_NATIVE_TRACE_GRAPH_CLOCK=1 (default OFF, opt-in until
  // validated).
  unsigned long long* clock_buf_device_ = nullptr;  // device buffer N+1 u64
  unsigned long long* clock_buf_host_ = nullptr;    // host pinned mirror
  size_t clock_buf_slots_ = 0;
  std::vector<std::string> per_kernel_clock_names_;

  void capture(std::function<void(hipStream_t)> workload, hipStream_t stream);
  void exec(hipStream_t stream);
};

}  // namespace HugeCTR