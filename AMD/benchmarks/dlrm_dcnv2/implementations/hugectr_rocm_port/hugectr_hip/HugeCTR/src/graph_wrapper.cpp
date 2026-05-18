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
      // Register the chain with the emitter for harvest. Stream name is
      // "default" -- captured graphs typically replay on the compute stream.
      tracing::PerfettoEmitter::instance().register_graph_event_chain(
          dev, "default", graph_start_event_, per_kernel_events_,
          per_kernel_names_);
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