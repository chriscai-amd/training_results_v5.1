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

#include <atomic>
#include <common.hpp>
#include <cstdio>
#include <cstdlib>
#include <graph_clock_writer.hpp>
#include <graph_wrapper.hpp>
#include <map>
#include <mutex>
#include <perfetto_emitter.hpp>

namespace HugeCTR {

namespace {

// Phase 20.8: walk the captured graph and insert event-record nodes
// AFTER each kernel node. Returns vector of inserted events (in graph
// order). Plus inserts a single graph-start event as a root marker.
// Returns 0 on failure (graph stays unmodified).
static size_t insert_per_kernel_event_nodes(hipGraph_t graph,
                                            hipEvent_t& graph_start_event,
                                            std::vector<hipEvent_t>& events_out,
                                            std::vector<std::string>& names_out) {
  // 1. Enumerate all nodes.
  size_t num_nodes = 0;
  hipError_t e = hipGraphGetNodes(graph, nullptr, &num_nodes);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/graph] hipGraphGetNodes(count) failed: %s\n",
                 hipGetErrorString(e));
    return 0;
  }
  if (num_nodes == 0) return 0;
  std::vector<hipGraphNode_t> nodes(num_nodes);
  e = hipGraphGetNodes(graph, nodes.data(), &num_nodes);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/graph] hipGraphGetNodes(fill) failed: %s\n",
                 hipGetErrorString(e));
    return 0;
  }

  // 2. Filter to kernel nodes.
  std::vector<hipGraphNode_t> kernel_nodes;
  kernel_nodes.reserve(num_nodes);
  for (auto& n : nodes) {
    hipGraphNodeType t;
    if (hipGraphNodeGetType(n, &t) == hipSuccess && t == hipGraphNodeTypeKernel) {
      kernel_nodes.push_back(n);
    }
  }
  if (kernel_nodes.empty()) return 0;

  // 3. Insert graph-start event (no deps -> becomes a root marker).
  e = hipEventCreate(&graph_start_event);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/graph] hipEventCreate(start) failed: %s\n",
                 hipGetErrorString(e));
    return 0;
  }
  hipGraphNode_t start_node;
  e = hipGraphAddEventRecordNode(&start_node, graph, nullptr, 0, graph_start_event);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/graph] hipGraphAddEventRecordNode(start) failed: %s -- "
                        "ROCm 7.2.1 may not support post-capture event insertion. "
                        "Skipping per-kernel event nodes.\n",
                 hipGetErrorString(e));
    hipEventDestroy(graph_start_event);
    graph_start_event = nullptr;
    return 0;
  }

  // 4. For each kernel node, insert event AFTER it (deps = {kernel_node}).
  // No edge modification: K already has its captured downstream edges.
  // Our event-record node is an additional child of K. The event fires
  // when K completes. Subsequent kernel K' (also captured-child of K)
  // starts when K completes too; they execute in parallel on the same
  // stream but the GPU clock readings are still correct for duration
  // computation: duration(K') = elapsedTime(event_after_K, event_after_K').
  events_out.reserve(kernel_nodes.size());
  names_out.reserve(kernel_nodes.size());
  for (size_t i = 0; i < kernel_nodes.size(); ++i) {
    hipEvent_t ev;
    e = hipEventCreate(&ev);
    if (e != hipSuccess) {
      std::fprintf(stderr,
                   "[HCTR perfetto/graph] hipEventCreate(%zu) failed: %s\n",
                   i, hipGetErrorString(e));
      break;
    }
    hipGraphNode_t event_node;
    e = hipGraphAddEventRecordNode(&event_node, graph, &kernel_nodes[i], 1, ev);
    if (e != hipSuccess) {
      std::fprintf(stderr,
                   "[HCTR perfetto/graph] hipGraphAddEventRecordNode(%zu) failed: %s\n",
                   i, hipGetErrorString(e));
      hipEventDestroy(ev);
      break;
    }
    events_out.push_back(ev);
    char nm[40];
    std::snprintf(nm, sizeof(nm), "g/kernel_%zu", i);
    names_out.emplace_back(nm);
  }
  return events_out.size();
}

inline bool graph_nodes_trace_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_GRAPH_NODES");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

inline bool graph_clock_trace_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_GRAPH_CLOCK");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Phase 20.9: insert clock-writer KERNEL nodes after every kernel node
// in the captured graph (plus one root marker for iter-start). Returns
// the count of slots inserted (N+1). 0 on failure.
//
// Kernel nodes work reliably in HIP graphs (unlike event-record nodes
// which return 0us elapsedTime on ROCm 7.2.1). The writer kernel does
// a single 64-bit global store of wall_clock64(), executes in ~1us.
static size_t insert_per_kernel_clock_nodes(
    hipGraph_t graph,
    unsigned long long*& clock_buf_device,
    unsigned long long*& clock_buf_host,
    size_t& clock_buf_slots,
    std::vector<std::string>& names_out) {
  size_t num_nodes = 0;
  if (hipGraphGetNodes(graph, nullptr, &num_nodes) != hipSuccess) return 0;
  std::vector<hipGraphNode_t> nodes(num_nodes);
  if (hipGraphGetNodes(graph, nodes.data(), &num_nodes) != hipSuccess) return 0;

  std::vector<hipGraphNode_t> kernel_nodes;
  for (auto& n : nodes) {
    hipGraphNodeType t;
    if (hipGraphNodeGetType(n, &t) == hipSuccess && t == hipGraphNodeTypeKernel) {
      kernel_nodes.push_back(n);
    }
  }
  if (kernel_nodes.empty()) return 0;

  // Allocate device + pinned host buffer for N+1 u64 slots.
  size_t slots = kernel_nodes.size() + 1;  // +1 for graph-start marker
  size_t bytes = slots * sizeof(unsigned long long);
  hipError_t e = hipMalloc(&clock_buf_device, bytes);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/clock] hipMalloc failed: %s\n",
                 hipGetErrorString(e));
    return 0;
  }
  hipMemset(clock_buf_device, 0, bytes);
  e = hipHostMalloc(reinterpret_cast<void**>(&clock_buf_host), bytes,
                    hipHostMallocDefault);
  if (e != hipSuccess) {
    std::fprintf(stderr, "[HCTR perfetto/clock] hipHostMalloc failed: %s\n",
                 hipGetErrorString(e));
    hipFree(clock_buf_device);
    clock_buf_device = nullptr;
    return 0;
  }
  clock_buf_slots = slots;

  // Add 1 clock-writer node at the start (no deps, becomes a root).
  // Then N writer nodes, each depending on the corresponding kernel.
  auto add_writer = [&](int slot, hipGraphNode_t* dep, size_t ndep,
                        hipGraphNode_t* out_node) -> hipError_t {
    // hipKernelNodeParams: func, gridDim, blockDim, sharedMem, kernelParams
    hipKernelNodeParams p = {};
    p.func = (void*)tracing::graph_clock_writer_kernel;
    p.gridDim = dim3(1, 1, 1);
    p.blockDim = dim3(1, 1, 1);
    p.sharedMemBytes = 0;
    static thread_local unsigned long long* arg0;
    static thread_local int arg1;
    arg0 = clock_buf_device;
    arg1 = slot;
    void* kp[2] = {(void*)&arg0, (void*)&arg1};
    p.kernelParams = kp;
    p.extra = nullptr;
    return hipGraphAddKernelNode(out_node, graph, dep, ndep, &p);
  };

  hipGraphNode_t start_node;
  e = add_writer(0, nullptr, 0, &start_node);
  if (e != hipSuccess) {
    std::fprintf(stderr,
                 "[HCTR perfetto/clock] add start writer failed: %s -- "
                 "ROCm 7.2.1 may not support hipGraphAddKernelNode "
                 "post-capture insertion either; falling back.\n",
                 hipGetErrorString(e));
    hipFree(clock_buf_device);
    hipHostFree(clock_buf_host);
    clock_buf_device = nullptr;
    clock_buf_host = nullptr;
    clock_buf_slots = 0;
    return 0;
  }
  names_out.reserve(kernel_nodes.size());
  for (size_t i = 0; i < kernel_nodes.size(); ++i) {
    hipGraphNode_t writer_node;
    e = add_writer(static_cast<int>(i + 1), &kernel_nodes[i], 1, &writer_node);
    if (e != hipSuccess) {
      std::fprintf(stderr,
                   "[HCTR perfetto/clock] add writer %zu failed: %s\n",
                   i, hipGetErrorString(e));
      break;
    }
    char nm[40];
    std::snprintf(nm, sizeof(nm), "[graph] kernel_%zu", i);
    names_out.emplace_back(nm);
  }
  return names_out.size() + 1;  // +1 for the start marker
}

}  // namespace

void GraphWrapper::capture(std::function<void(hipStream_t)> workload, hipStream_t stream) {
  if (initialized) {
    return;
  }

  HCTR_LIB_THROW(hipStreamBeginCapture(stream, hipStreamCaptureModeThreadLocal));
  workload(stream);
  HCTR_LIB_THROW(hipStreamEndCapture(stream, &graph));

  // Phase 20.8: optional per-kernel event-record nodes inserted into
  // the captured graph BEFORE instantiation. Adds ~zero per-replay
  // host overhead (events execute as graph nodes). On ROCm 7.2.1 this
  // may fail with the known event-in-graph bug -- if so, we log to
  // stderr and proceed without per-kernel events; graph still works.
  if (graph_nodes_trace_enabled()) {
    int dev = 0;
    hipGetDevice(&dev);
    per_kernel_rank_ = dev;
    size_t inserted = insert_per_kernel_event_nodes(
        graph, graph_start_event_, per_kernel_events_, per_kernel_names_);
    if (inserted > 0) {
      std::fprintf(stderr,
                   "[HCTR perfetto/graph] rank %d: inserted %zu per-kernel "
                   "event-record nodes into captured graph (no per-replay "
                   "host overhead)\n",
                   dev, inserted);
      tracing::PerfettoEmitter::instance().register_graph_event_chain(
          dev, "default", graph_start_event_, per_kernel_events_,
          per_kernel_names_);
    }
  }

  // Phase 20.9: clock-writer kernel approach -- workaround for the ROCm
  // event-in-graph bug. Inserts kernel nodes (which DO work in graphs)
  // that each write wall_clock64() to a buffer slot. Harvested on host
  // post-replay; durations come from clock-tick deltas.
  if (graph_clock_trace_enabled()) {
    int dev = 0;
    hipGetDevice(&dev);
    per_kernel_rank_ = dev;
    size_t slots = insert_per_kernel_clock_nodes(
        graph, clock_buf_device_, clock_buf_host_, clock_buf_slots_,
        per_kernel_clock_names_);
    if (slots > 0) {
      std::fprintf(stderr,
                   "[HCTR perfetto/clock] rank %d: inserted %zu clock-writer "
                   "kernel nodes (~1us per node, zero per-replay host cost)\n",
                   dev, slots);
      tracing::PerfettoEmitter::instance().register_graph_clock_chain(
          dev, "default", clock_buf_device_, clock_buf_host_, clock_buf_slots_,
          per_kernel_clock_names_);
    }
  }

  HCTR_LIB_THROW(hipGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));
  initialized = true;
}

void GraphWrapper::exec(hipStream_t stream) {
  if (!initialized) {
    HCTR_OWN_THROW(Error_t::IllegalCall, "Trying to execute graph which was not captured");
  }
  HCTR_LIB_THROW(hipGraphLaunch(graph_exec, stream));
}

}  // namespace HugeCTR