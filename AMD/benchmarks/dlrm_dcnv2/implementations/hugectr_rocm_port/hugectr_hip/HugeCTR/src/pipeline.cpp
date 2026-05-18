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

#include <unistd.h>

#include <hctr_tracing.hpp>
#include <perfetto_emitter.hpp>
#include <pipeline.hpp>

namespace HugeCTR {

StreamContextScheduleable::StreamContextScheduleable(std::function<void()> workload)
    : stream_name_(std::nullopt),
      priority_(0),
      is_absolute_stream_(false),
      schedule_event_(std::nullopt),
      wait_external_(false),
      completion_event_(std::nullopt),
      record_external_(false),
      workload_(workload) {}

StreamContextScheduleable::~StreamContextScheduleable() {
  if (completion_event_) {
    hipEventDestroy(completion_event_.value());
  }
}

void StreamContextScheduleable::set_absolute_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = true;
}

void StreamContextScheduleable::set_stream(const std::string &stream_name, int priority) {
  stream_name_ = stream_name;
  priority_ = priority;
  is_absolute_stream_ = false;
}

std::tuple<std::string, int> StreamContextScheduleable::get_stream_name(
    std::shared_ptr<GPUResource> gpu) {
  return {is_absolute_stream_ ? stream_name_.value_or("")
                              : gpu->get_current_stream_name() + stream_name_.value_or(""),
          priority_};
}

void StreamContextScheduleable::wait_event(const std::vector<hipEvent_t> &schedule_event,
                                           bool external) {
  HCTR_CHECK_HINT(!schedule_event_, "duplicate wait_event.");
  schedule_event_ = schedule_event;
  wait_external_ = external;
}

hipEvent_t StreamContextScheduleable::record_done(bool external, unsigned int flags) {
  if (!completion_event_) {
    hipEvent_t event;
    // Phase 14s (perf-push, overlap-fix #1, hypothesis 2): when External
    // wait/record is restored, force events to be created with
    // hipEventDefault (no DisableTiming). ROCm 7.2.1 std::bad_alloc may
    // come from incompatibility between hipEventDisableTiming and
    // hipEventWaitExternal: the runtime may try to allocate timer-state
    // proportional to graph nodes when DisableTiming is set on an event
    // referenced by an External wait.
    static const bool kRestoreExternal = []() {
      const char* e = std::getenv("HCTR_RESTORE_WAIT_EXTERNAL");
      return e != nullptr && e[0] == '1';
    }();
    unsigned int actual_flags = (kRestoreExternal && external) ? hipEventDefault : flags;
    HCTR_LIB_THROW(hipEventCreateWithFlags(&event, actual_flags));
    completion_event_ = event;
    record_external_ = external;
  }
  HCTR_CHECK_HINT(external == record_external_, "contradictory record_done.");
  return completion_event_.value();
}

void StreamContextScheduleable::init(std::shared_ptr<GPUResource> gpu) {
  CudaDeviceContext context{gpu->get_device_id()};

  auto [current_stream_name, priority] = get_stream_name(gpu);
  StreamContext stream_context{gpu, current_stream_name, priority};
}

void StreamContextScheduleable::run(std::shared_ptr<GPUResource> gpu, bool use_graph) {
  // Phase 20 (2026-05-16): tag the entire run body (event waits +
  // workload + record) with a ROCTX range named by set_debug_name().
  // Free when HCTR_ROCTX is unset.
  tracing::ScopedRange _hctr_roctx_scope(debug_name_.c_str());

  CudaDeviceContext context{gpu->get_device_id()};

  // Phase 20.4 (2026-05-17): lazily register a GpuPhase for this
  // Scheduleable on first run (HCTR_NATIVE_TRACE=1 only). Must happen
  // AFTER CudaDeviceContext so hipEventCreate() lands on the correct GPU.
  // Per-scheduleable phase recording can be independently disabled via
  // HCTR_NATIVE_TRACE_NO_PHASES=1 (for bisecting crashes -- keeps
  // begin_iter/end_iter active but no per-phase event records).
  static const bool no_phases = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_NO_PHASES");
    return e != nullptr && e[0] == '1';
  }();
  if (!no_phases && tracing::native_trace_enabled() &&
      gpu_phase_ == nullptr && !debug_name_.empty()) {
    gpu_phase_ = tracing::PerfettoEmitter::instance().register_phase(
        gpu->get_local_id(), debug_name_);
  }

  auto [current_stream_name, priority] = get_stream_name(gpu);
  StreamContext stream_context{gpu, current_stream_name, priority};
  hipStream_t stream = gpu->get_stream();
  if (schedule_event_.has_value()) {
    for (hipEvent_t event : schedule_event_.value()) {
      // Phase 14s (perf-push, overlap-fix #1): restore symmetry with
      // record_external_ (line 104, kept as hipEventRecordExternal). Phase
      // 14p stripped the wait side to 0:0 to dodge a std::bad_alloc on
      // ROCm 7.2.1, but NV upstream pipeline.cpp:91-95 ALWAYS honors
      // wait_external_, and without it the captured-graph cross-stream
      // dependencies (dp_stream / mp_stream) collapse to intra-graph
      // nodes, serializing onto the launch stream. Phase 14r 8-GPU trace:
      // AMD 24% overlap vs NV 87% directly stems from this.
      // Gated behind HCTR_RESTORE_WAIT_EXTERNAL=1 (default OFF) for
      // safe A/B; also independently A/B on ROCm 7.12 since the 7.2
      // std::bad_alloc may be a runtime bug fixed in 7.12.
      static const bool kRestoreWaitExternal = []() {
        const char* env = std::getenv("HCTR_RESTORE_WAIT_EXTERNAL");
        return env != nullptr && env[0] == '1';
      }();
      HCTR_LIB_THROW(hipStreamWaitEvent(
          stream, event,
          (kRestoreWaitExternal && wait_external_ && use_graph)
              ? hipEventWaitExternal : 0));
    }
  }
  // Phase 20.4: bracket workload with start/end hipEventRecords.
  // SKIP recording when the stream is currently in HIP graph capture mode:
  // recording timing-enabled events into a captured graph triggers a ROCm
  // 7.2.1 runtime bug (similar root cause as AddExternalEventWaitNode type
  // confusion). The outer GraphScheduleable still records per-graph timing
  // on the non-captured stream BEFORE/AFTER graph_.exec(). Standalone
  // StreamContextScheduleables (data_distribute, ebc_*, comm, opt) are NOT
  // in capture mode so they DO get per-phase timing.
  hipStreamCaptureStatus cap_status = hipStreamCaptureStatusNone;
  if (gpu_phase_ != nullptr) {
    hipStreamIsCapturing(stream, &cap_status);
  }
  {
    tracing::GpuPhase* phase_for_this_run =
        (cap_status == hipStreamCaptureStatusActive) ? nullptr : gpu_phase_;
    tracing::ScopedGpuPhase _hctr_gpu_phase(phase_for_this_run, stream,
                                            current_stream_name);
    if (workload_) workload_();
  }
  if (completion_event_.has_value()) {
    HCTR_LIB_THROW(hipEventRecordWithFlags(
        completion_event_.value(), stream,
        record_external_ && use_graph ? hipEventRecordExternal : hipEventRecordDefault));
  }
}

void GraphScheduleable::run(std::shared_ptr<GPUResource> gpu, bool use_graph) {
  if (scheduleable_list_.empty()) return;
  // Phase 20 (2026-05-16): ROCTX range around the entire graph run.
  // Each inner StreamContextScheduleable emits its own nested range.
  std::string outer_name = debug_name_.empty() ? std::string("graph") : "graph_" + debug_name_;
  tracing::ScopedRange _hctr_graph_roctx_scope(outer_name.c_str());

  // Phase 20.4 deep-trace mode: when HCTR_NATIVE_TRACE_DEEP=1, force
  // use_graph=false for this GraphScheduleable so its inner scheduleables
  // run as individual stream-context scheduleables. This lets us record
  // per-inner-scheduleable GPU phases (fwd/bmlp, fwd/tmlp, fwd/loss,
  // bwd/tmlp, bwd/bmlp, etc.) which are otherwise hidden inside the
  // captured-graph black box. Cost: ~30% iter-wall overhead since the
  // network kernels lose graph-launch fusion benefits.
  static const bool deep_trace = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_DEEP");
    return e != nullptr && e[0] == '1';
  }();
  if (deep_trace && tracing::native_trace_enabled()) {
    use_graph = false;
  }

  auto do_it = [=](hipStream_t) {
    for (auto &scheduleable : scheduleable_list_) {
      scheduleable->run(gpu, use_graph);
    }
  };
  auto first_node = std::dynamic_pointer_cast<StreamContextScheduleable>(scheduleable_list_[0]);
  HCTR_THROW_IF(first_node == nullptr, Error_t::WrongInput,
                "first node of GraphSchedule should be StreamScheduleable");
  auto [current_stream_name, priority] = first_node->get_stream_name(gpu);
  hipStream_t stream = gpu->get_stream(current_stream_name, priority);

  // Phase 20.4: lazily register a phase for the graph's outer range.
  // (Gated by HCTR_NATIVE_TRACE_NO_PHASES for bisect.)
  // Phase 20.9c: use NV-style "[graph] name" prefix so Perfetto buckets
  // this in a "graph" cat (was unprefixed "graph_<name>" -> "phase" cat).
  static const bool no_phases_g = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_NO_PHASES");
    return e != nullptr && e[0] == '1';
  }();
  if (!no_phases_g && tracing::native_trace_enabled() &&
      gpu_phase_ == nullptr && !debug_name_.empty()) {
    CudaDeviceContext _dctx{gpu->get_device_id()};
    gpu_phase_ = tracing::PerfettoEmitter::instance().register_phase(
        gpu->get_local_id(), "[graph] " + debug_name_);
  }

  // Phase 14n.5: cross-graph wait_event before replay. Captured cross-stream
  // edge ensures bprop graph (on stream 2) waits for fprop graph (on default).
  if (wait_external_event_.has_value()) {
    for (hipEvent_t event : wait_external_event_.value()) {
      HCTR_LIB_THROW(hipStreamWaitEvent(stream, event, 0));
    }
  }

  if (!use_graph) {
    {
      tracing::ScopedGpuPhase _hctr_gpu_phase(gpu_phase_, stream, current_stream_name);
      do_it(stream);
    }
    if (completion_event_.has_value()) {
      HCTR_LIB_THROW(hipEventRecord(completion_event_.value(), stream));
    }
    return;
  }
  if (!graph_.initialized) {
    graph_.capture(do_it, stream);
#ifdef ENABLE_MPI
#pragma omp master
    MPI_Barrier(MPI_COMM_WORLD);
#endif
#pragma omp barrier
  }
  {
    tracing::ScopedGpuPhase _hctr_gpu_phase(gpu_phase_, stream, current_stream_name);
    graph_.exec(stream);
  }
  // Phase 14n.5: record completion event AFTER replay so downstream graphs/scheduleables
  // can wait_event on this graph's completion. Note: hipEventRecord on a stream after
  // graph_.exec() gives semantics "this event signals when all prior work on stream is done"
  // which includes the replayed graph.
  if (completion_event_.has_value()) {
    HCTR_LIB_THROW(hipEventRecord(completion_event_.value(), stream));
  }
}

Pipeline::Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
                   const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list)
    : stream_name_(stream_name),
      gpu_resource_(std::move(gpu_resource)),
      scheduleable_list_(scheduleable_list) {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->init(gpu_resource_);
  }
}

void Pipeline::run() {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, false);
  }
}

void Pipeline::run_graph() {
  StreamContext stream_context(gpu_resource_, stream_name_);
  for (auto &scheduleable : scheduleable_list_) {
    scheduleable->run(gpu_resource_, true);
  }
}

}  // namespace HugeCTR