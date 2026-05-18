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

#include <common.hpp>
#include <functional>
#include <gpu_resource.hpp>
#include <graph_wrapper.hpp>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace HugeCTR {

namespace tracing {
class GpuPhase;  // forward decl; full def in perfetto_emitter.hpp
}  // namespace tracing

class Scheduleable {
 public:
  virtual ~Scheduleable() = default;

  virtual void init(std::shared_ptr<GPUResource> gpu){};

  virtual void run(std::shared_ptr<GPUResource> gpu, bool use_graph) = 0;
};

class StreamContextScheduleable : public Scheduleable {
 private:
  std::optional<std::string> stream_name_;
  int priority_;
  bool is_absolute_stream_;
  std::optional<std::vector<hipEvent_t>> schedule_event_;
  bool wait_external_;
  std::optional<hipEvent_t> completion_event_;
  bool record_external_;
  // Phase 20 (2026-05-16): optional human-readable name for ROCTX
  // tagging in traces (HCTR_ROCTX=1). Set via set_debug_name().
  std::string debug_name_;
  // Phase 20.4 (2026-05-17): per-Scheduleable GPU phase for HCTR-native
  // Perfetto emitter (HCTR_NATIVE_TRACE=1). Lazily created on first run().
  // Non-owning: lifetime managed by PerfettoEmitter singleton.
  tracing::GpuPhase* gpu_phase_ = nullptr;

  std::function<void()> workload_;

 public:
  HCTR_DISALLOW_COPY_AND_MOVE(StreamContextScheduleable);

  explicit StreamContextScheduleable(std::function<void()> workload);

  ~StreamContextScheduleable() override;

  void set_absolute_stream(const std::string &stream_name, int priority = 0);

  void set_stream(const std::string &stream_name, int priority = 0);

  // Phase 20: tag this scheduleable so HCTR_ROCTX=1 traces show a
  // host-side range named `name` around the workload + its kernel
  // launches.
  void set_debug_name(std::string name) { debug_name_ = std::move(name); }
  const std::string &debug_name() const { return debug_name_; }

  std::tuple<std::string, int> get_stream_name(std::shared_ptr<GPUResource> gpu);

  void wait_event(const std::vector<hipEvent_t> &schedule_event, bool external = false);

  hipEvent_t record_done(bool external = false, unsigned int flags = hipEventDisableTiming);

  void init(std::shared_ptr<GPUResource> gpu) override;

  void run(std::shared_ptr<GPUResource> gpu, bool use_graph) override;
};

class GraphScheduleable : public Scheduleable {
 private:
  std::vector<std::shared_ptr<Scheduleable>> scheduleable_list_;
  GraphWrapper graph_;

  // Phase 14n.5 (2026-05-14): cross-graph sync events. wait_external_event_
  // is set via wait_event(); the run() method records hipStreamWaitEvent(stream, event)
  // BEFORE replaying the graph. completion_event_ is created on demand by record_done()
  // and recorded AFTER graph replay so other Scheduleables can wait on it.
  std::optional<std::vector<hipEvent_t>> wait_external_event_;
  std::optional<hipEvent_t> completion_event_;

  // Phase 20 (2026-05-16): optional human-readable name for ROCTX tagging
  // in traces (HCTR_ROCTX=1). Set via set_debug_name().
  std::string debug_name_;
  // Phase 20.4 (2026-05-17): GPU phase for the entire graph replay.
  tracing::GpuPhase* gpu_phase_ = nullptr;

 public:
  HCTR_DISALLOW_COPY_AND_MOVE(GraphScheduleable);

  template <typename... T>
  GraphScheduleable(std::shared_ptr<T>... scheduleable) {
    static_assert(std::conjunction<std::is_base_of<Scheduleable, T>...>::value, "");
    (scheduleable_list_.push_back(std::dynamic_pointer_cast<Scheduleable>(scheduleable)), ...);
  }

  GraphScheduleable(std::vector<std::shared_ptr<Scheduleable>> scheduleable_list)
      : scheduleable_list_(scheduleable_list) {}

  ~GraphScheduleable() override {
    if (completion_event_.has_value()) {
      hipEventDestroy(completion_event_.value());
    }
  }

  // Phase 20: tag this graph so HCTR_ROCTX=1 traces show a host-side
  // range named graph_<name> around capture + replay.
  void set_debug_name(std::string name) { debug_name_ = std::move(name); }
  const std::string& debug_name() const { return debug_name_; }

  // Make this graph wait for externally-recorded events BEFORE its replay starts.
  // Used to express cross-graph dependencies (e.g., bprop graph waits for fprop graph).
  void wait_event(const std::vector<hipEvent_t>& events) { wait_external_event_ = events; }

  // Get/create a completion event recorded AFTER this graph's replay finishes.
  // Returned event can be passed to other Scheduleables' wait_event().
  hipEvent_t record_done() {
    if (!completion_event_.has_value()) {
      hipEvent_t ev;
      HCTR_LIB_THROW(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
      completion_event_ = ev;
    }
    return completion_event_.value();
  }

  void run(std::shared_ptr<GPUResource> gpu, bool use_graph) override;
};

class Pipeline {
 private:
  std::string stream_name_;
  std::shared_ptr<GPUResource> gpu_resource_;
  std::vector<std::shared_ptr<Scheduleable>> scheduleable_list_;

 public:
  Pipeline() = default;

  Pipeline(const std::string &stream_name, std::shared_ptr<GPUResource> gpu_resource,
           const std::vector<std::shared_ptr<Scheduleable>> &scheduleable_list);

  std::string get_stream_name() { return stream_name_; }

  void run();

  void run_graph();
};
}  // namespace HugeCTR