/*
 * Phase 20.4 (2026-05-17): HCTR-native Perfetto emitter.
 *
 * Goal: produce a Chrome Trace Format (Perfetto-compatible) JSON of the
 * training iteration at near-PRODUCTION wall time (<1% overhead), with
 * per-stream lanes and per-phase events, matching the fidelity of an
 * NV nsys/CUPTI trace.
 *
 * Why not rocprofv3:
 *   rocprofv3 --kernel-trace adds ~7 ms/iter on AMD because it tracks
 *   every HSA kernel dispatch with completion callbacks. It is not
 *   HIP-Graph-aware -- each captured node is treated as a fresh dispatch.
 *   The captured iter therefore inflates from ~4 ms to ~12 ms.
 *
 * How this works:
 *   - Each StreamContextScheduleable / GraphScheduleable owns a GpuPhase.
 *   - GpuPhase holds a pair of timing-enabled hipEvents (start, end).
 *   - In run(), we hipEventRecord start before the workload and end after.
 *   - Inside captured graphs, these become graph nodes that get re-recorded
 *     on each replay -- giving GPU-clock-accurate per-iter timings.
 *   - At iter end, harvest_iter() reads all phase elapsed times via
 *     hipEventElapsedTime() and appends to an in-memory event buffer.
 *   - At training end (or every N iters), flush() writes the buffer to a
 *     per-rank Chrome Trace Format JSON file.
 *
 * Default OFF (no overhead). Enable with HCTR_NATIVE_TRACE=1 env var.
 *
 * Output: ${HCTR_NATIVE_TRACE_DIR:-/tmp}/hctr_native_trace_rank${RANK}.json
 *         (drag into ui.perfetto.dev)
 */
#pragma once

#include <hip/hip_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace HugeCTR {
namespace tracing {

// Cached env-gate. HCTR_NATIVE_TRACE=1 enables the whole subsystem.
// When disabled, all functions in this module are no-ops with branch-free
// inline checks at the call site.
inline bool native_trace_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// Phase 20.7 (2026-05-18): tiered detail knob. HCTR_NATIVE_TRACE_DETAIL
// selects fidelity vs overhead trade-off. All levels give ACCURATE
// per-op latencies -- the lower levels just record FEWER events:
//   0 = top-level scheduleable phases only (~26 evs/iter, +4% overhead)
//   1 = +L/* per-layer wrappers in Network::prop_layers (~50 evs/iter, +15%)
//   2 = +fc/* per-GEMM + cross/* per-cross-layer (~110 evs/iter, +50%) [DEFAULT]
// Each hipEventRecord costs ~4-5us on AMD (host-side), so event count
// directly drives overhead. Use lower levels for production-near
// snapshots; level 2 for NV-parity diagnostics.
inline int native_trace_detail() {
  static const int level = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_DETAIL");
    if (e == nullptr || e[0] == '\0') return 2;  // default: full
    int v = std::atoi(e);
    if (v < 0) v = 0;
    if (v > 2) v = 2;
    return v;
  }();
  return level;
}

// A single named GPU phase (one per Scheduleable). Owns a pair of
// timing-enabled hipEvents that are recorded around the workload.
// Thread-safety: methods are NOT thread-safe; each GpuPhase is intended
// to be touched only by the single thread that runs its owning Scheduleable.
class GpuPhase {
 public:
  // name: phase name (e.g., "fwd/bmlp"). Will appear in Perfetto.
  // rank: GPU rank (used for pid in Chrome Trace Format).
  GpuPhase(int rank, std::string name);
  ~GpuPhase();

  GpuPhase(const GpuPhase&) = delete;
  GpuPhase& operator=(const GpuPhase&) = delete;

  // Record start event on the given stream. Called at the top of run().
  void record_start(hipStream_t stream, const std::string& stream_name);

  // Record end event on the given stream. Called at the bottom of run().
  void record_end(hipStream_t stream);

  // Try to read elapsed-time data into the emitter's buffer. Called by
  // PerfettoEmitter::harvest_iter() at iter end. Returns true if data
  // was harvested for this iter.
  //
  // iter_base_event: a reference event recorded at start of iter on the
  //   compute stream. Used to compute offset within the iter.
  // iter_idx: monotonic iter counter (for cat tagging).
  // iter_start_ns: host-side iter start timestamp (for absolute positioning).
  bool try_harvest(int iter_idx, hipEvent_t iter_base_event, int64_t iter_start_ns);

  // Mark the phase as "fresh data available this iter". Called by
  // PerfettoEmitter::begin_iter for all registered phases (since
  // captured-graph phases re-record events on every replay without
  // calling record_start).
  void mark_active() { recorded_this_iter_ = true; }
  // Phase 20.9g: called by PerfettoEmitter::begin_iter to clear stale
  // record flags from warmup iters. See begin_iter for full rationale.
  void mark_unrecorded() { recorded_this_iter_ = false; }

  const std::string& name() const { return name_; }

  // Phase 20.7 (2026-05-18): set static per-phase args metadata that
  // will be merged into every Perfetto event emitted for this phase.
  // Pass a JSON fragment WITHOUT outer braces, e.g.
  //   set_args_json(R"("M":4096,"N":512,"K":256,"dtype":"fp16")");
  // The string is copied and stored once; emit_event() inlines it
  // into the per-event args block (alongside the iter index).
  // Useful for: GEMM shapes, kernel grid/block, kernel demangled name.
  void set_args_json(std::string args) { args_json_ = std::move(args); }
  const std::string& args_json() const { return args_json_; }

 private:
  int rank_;
  std::string name_;
  // The stream name captured on the FIRST record_start call -- since most
  // Scheduleables run on a fixed stream, this is stable for the lifetime
  // of the phase.
  std::string stream_name_;
  hipEvent_t start_event_;
  hipEvent_t end_event_;
  bool recorded_this_iter_;
  // Phase 20.7: static metadata (JSON fragment) merged into args on emit.
  std::string args_json_;
};

// A single Perfetto event entry to be written to JSON. Stored in a
// flat host-side buffer by harvest_iter(). Memory layout: ~80 bytes / event.
struct PerfettoEvent {
  int64_t ts_us;  // absolute timestamp in microseconds since epoch
  double dur_us;  // duration in microseconds
  int rank;       // -> pid in JSON
  std::string stream_name;  // -> tid string in JSON (Perfetto allows string tids)
  std::string name;
  std::string cat;
  int iter_idx;
  // Phase 20.7: PyTorch-profiler-style args metadata (JSON fragment
  // WITHOUT outer braces, e.g. "M":4096,"N":512,"K":256). Inlined into
  // args alongside iter index. Empty means no extra args.
  std::string args_extra;
};

// Singleton: owns all registered phases + the event buffer.
// Created on first call to instance(); destructor flushes to disk.
class PerfettoEmitter {
 public:
  static PerfettoEmitter& instance();

  // Register a phase. Returns a non-owning pointer (lifetime = emitter).
  // Called once per Scheduleable at construction or first run.
  GpuPhase* register_phase(int rank, const std::string& name);

  // Phase 20.7: register a phase with static args metadata
  // (PyTorch-profiler-style: kernel demangled name, GEMM shapes,
  // grid/block, etc.). The args_json fragment is merged into every
  // Perfetto event emitted for this phase.
  GpuPhase* register_phase(int rank, const std::string& name,
                            const std::string& args_json);

  // Phase 20.8: register a chained event-record sequence for a captured
  // HIP graph. Each event is recorded AFTER its corresponding kernel by
  // a graph node (zero per-replay host overhead). Per-kernel duration
  // is computed as elapsedTime(prev_event, this_event) where prev_event
  // is the graph_start_event for the first kernel, then the previous
  // kernel's post-event thereafter.
  //
  // Called by GraphWrapper after capture, before instantiate. All events
  // must be timing-enabled hipEvents. Harvest happens at end_iter().
  void register_graph_event_chain(int rank, const std::string& stream_name,
                                   hipEvent_t graph_start_event,
                                   const std::vector<hipEvent_t>& events,
                                   const std::vector<std::string>& names);

  // Phase 20.9: register a clock-writer chain for a captured HIP graph.
  // Each slot in buf_device contains the wall_clock64() value at the
  // moment the corresponding clock-writer kernel node fired. Slot 0 is
  // the graph-start marker; slots 1..N are after kernel 0..N-1.
  // Per-kernel duration = (buf[i+1] - buf[i]) / clock_rate_khz * 1000 ns.
  //
  // Workaround for ROCm event-in-graph bug (rocm-systems#2380): kernel
  // nodes work reliably in graphs while event-record nodes do not.
  void register_graph_clock_chain(int rank, const std::string& stream_name,
                                   unsigned long long* buf_device,
                                   unsigned long long* buf_host,
                                   size_t num_slots,
                                   const std::vector<std::string>& names);

  // Record an iter-base event on the given stream and start a host-side
  // iter timer. Called from Model::train() at the start of each iter.
  void begin_iter(int rank, int iter_idx, hipStream_t base_stream);

  // Harvest all phases registered for this rank since begin_iter().
  // Called at end of train() body for the SAME rank.
  void end_iter(int rank);

  // Append a raw event (used for host-side ranges like "iter_N" itself).
  void emit_event(PerfettoEvent e);

  // Phase 20.9d: emit a HOST-SIDE event measured with std::chrono. No GPU
  // events involved -- pure host wall time. Used to measure host API call
  // durations like hipGraphLaunch, hipLaunchKernel, hipStreamSynchronize.
  // Lane name appears as a Perfetto "thread" within the rank's process.
  void emit_host_event(int rank, const std::string& lane_name,
                        const std::string& event_name,
                        int64_t ts_ns, int64_t dur_ns,
                        const std::string& args_extra = "");

  // Flush buffer to per-rank JSON file. Called from main thread at exit
  // or periodically. Thread-safe.
  void flush(int rank);

  // Stream-name resolver (optional). If not set, GpuPhase will pass the
  // stream pointer's hex string as the tid.
  // (Empty for v0 prototype -- we get stream names from the call site.)

 private:
  PerfettoEmitter();
  ~PerfettoEmitter();

  // Phase 20.8: graph-event chain registered post-capture by GraphWrapper.
  // Each event in `events` is recorded after a kernel node in the captured
  // graph. Duration of kernel i = elapsedTime(events[i-1], events[i])
  // (or elapsedTime(graph_start_event, events[0]) for i=0).
  struct GraphEventChain {
    int rank;
    std::string stream_name;
    hipEvent_t graph_start_event;
    std::vector<hipEvent_t> events;
    std::vector<std::string> names;
  };

  // Phase 20.9: graph clock-writer chain (workaround for the ROCm
  // event-in-graph bug). buf[0] = graph-start tick; buf[i+1] = tick
  // after kernel i. clock_rate_khz used to convert ticks to ns.
  struct GraphClockChain {
    int rank;
    std::string stream_name;
    unsigned long long* buf_device;
    unsigned long long* buf_host;  // pinned mirror
    size_t num_slots;              // num_kernels + 1
    std::vector<std::string> names;  // num_kernels entries (after kernel 0..N-1)
    double clock_rate_khz;
  };

  struct RankState {
    int iter_idx = -1;
    int64_t iter_start_ns = 0;
    hipEvent_t iter_base_event = nullptr;  // created lazily
    // All GpuPhases ever registered to this rank (lifetime = emitter).
    // Captured-graph phases only call record_start during capture iter,
    // so we track them once and harvest every iter via hipEventQuery.
    std::vector<GpuPhase*> registered_phases;
    // Phase 20.8: chained graph event sequences for this rank.
    std::vector<GraphEventChain> graph_chains;
    // Phase 20.9: clock-writer chains for this rank.
    std::vector<GraphClockChain> graph_clock_chains;
  };

  std::mutex mu_;
  std::vector<std::unique_ptr<GpuPhase>> all_phases_;
  std::vector<PerfettoEvent> events_;     // accumulator, flushed periodically
  std::vector<RankState> rank_state_;     // indexed by rank id
  std::atomic<bool> flushed_at_exit_{false};

  // Friend access for harvest plumbing.
  friend class GpuPhase;
};

// Phase 20.9d: RAII helper to measure HOST-side wall time of a code block
// and emit it as a Perfetto event. Use to time HIP API calls like
// hipGraphLaunch directly, bypassing rocprofv3's instrumentation
// overhead (which itself is 5-10x of the real host call).
//
//   void GraphScheduleable::run(...) {
//     {
//       ScopedHostTimer _t(rank, "host_api", "[host_api] hipGraphLaunch");
//       graph_.exec(stream);
//     }
//   }
class ScopedHostTimer {
 public:
  ScopedHostTimer(int rank, std::string lane, std::string name,
                  std::string args_extra = "");
  ~ScopedHostTimer();
  ScopedHostTimer(const ScopedHostTimer&) = delete;
  ScopedHostTimer& operator=(const ScopedHostTimer&) = delete;

 private:
  bool active_;
  int rank_;
  std::string lane_;
  std::string name_;
  std::string args_extra_;
  int64_t t0_ns_;
};

// RAII helper: record start in ctor, end in dtor. Use inside run() bodies.
//   void StreamContextScheduleable::run(...) {
//     ScopedGpuPhase _p(gpu_phase_, stream, stream_name);
//     workload_();
//   }
class ScopedGpuPhase {
 public:
  ScopedGpuPhase(GpuPhase* phase, hipStream_t stream, const std::string& stream_name)
      : phase_(phase), stream_(stream) {
    if (phase_) phase_->record_start(stream_, stream_name);
  }
  ~ScopedGpuPhase() {
    if (phase_) phase_->record_end(stream_);
  }
  ScopedGpuPhase(const ScopedGpuPhase&) = delete;
  ScopedGpuPhase& operator=(const ScopedGpuPhase&) = delete;

 private:
  GpuPhase* phase_;
  hipStream_t stream_;
};

}  // namespace tracing
}  // namespace HugeCTR
