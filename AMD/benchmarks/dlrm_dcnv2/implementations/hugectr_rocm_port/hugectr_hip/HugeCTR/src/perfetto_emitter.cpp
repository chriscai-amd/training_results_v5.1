/*
 * Phase 20.4 (2026-05-17): HCTR-native Perfetto emitter implementation.
 * See perfetto_emitter.hpp for design rationale.
 */
#include <perfetto_emitter.hpp>

#include <hctr_tracing.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cxxabi.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace HugeCTR {
namespace tracing {

// ---------------------------------------------------------------------------
// Local helpers
// ---------------------------------------------------------------------------

namespace {

int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

// JSON-escape a string. Minimal -- handles backslash and quote.
std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 4);
  for (char c : s) {
    if (c == '"' || c == '\\') {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out.append("\\n");
    } else if (c == '\t') {
      out.append("\\t");
    } else {
      out.push_back(c);
    }
  }
  return out;
}

// Phase 20.10 (Path D, 2026-05-20): demangle a C++ ABI symbol with
// abi::__cxa_demangle. Returns the original mangled name on failure.
std::string demangle_symbol(const char* mangled) {
  if (mangled == nullptr || mangled[0] == '\0') return {};
  int status = 0;
  char* dem = abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
  std::string out;
  if (status == 0 && dem != nullptr) {
    out.assign(dem);
  } else {
    out.assign(mangled);
  }
  std::free(dem);
  return out;
}

}  // namespace

// Public: exposed in the header for callers that need to build their own
// args_json fragments with strings that may contain quotes / backslashes.
std::string escape_for_args_json(const std::string& s) {
  return json_escape(s);
}

// Public: resolve and cache a kernel function pointer to its demangled
// symbol name. See header for full contract.
std::string kernel_name_from_ptr(const void* func_ptr) {
  if (func_ptr == nullptr) return {};

  // Process-local cache: pointer-identity is stable for the lifetime of
  // the program (kernel symbols live in the loaded image), and the same
  // function pointer is reused across all dispatches of a given kernel.
  static std::mutex cache_mu;
  static std::unordered_map<const void*, std::string> cache;
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    auto it = cache.find(func_ptr);
    if (it != cache.end()) return it->second;
  }

  // hipKernelNameRefByPtr is available in ROCm 5.4+. The second arg is
  // the stream from which to derive the device context for kernel
  // symbol resolution; nullptr means "use the current device", which
  // is what we want for cache-on-first-launch semantics.
  const char* raw = nullptr;
#if defined(__HIP_PLATFORM_AMD__) || defined(HIP_VERSION)
  raw = hipKernelNameRefByPtr(func_ptr, /*stream=*/nullptr);
#endif
  std::string demangled = demangle_symbol(raw);

  std::lock_guard<std::mutex> lk(cache_mu);
  cache.emplace(func_ptr, demangled);
  return demangled;
}

namespace {

// Resolve the trace output directory. Defaults to /tmp.
std::string trace_dir() {
  const char* d = std::getenv("HCTR_NATIVE_TRACE_DIR");
  return (d && d[0]) ? std::string(d) : std::string("/tmp");
}

}  // namespace

// ---------------------------------------------------------------------------
// GpuPhase
// ---------------------------------------------------------------------------

GpuPhase::GpuPhase(int rank, std::string name)
    : rank_(rank),
      name_(std::move(name)),
      start_event_(nullptr),
      end_event_(nullptr),
      recorded_this_iter_(false) {
  // Phase 20.9 (2026-05-18): TESTED hipEventDisableSystemFence (0x4)
  // for lower per-record host cost -- causes "invalid resource handle"
  // crash in unrelated kernels on ROCm 7.2.1. Reverted to hipEventDefault.
  // The flag MAY work on newer ROCm; auto-tests when next version lands.
  hipError_t e1 = hipEventCreateWithFlags(&start_event_, hipEventDefault);
  hipError_t e2 = hipEventCreateWithFlags(&end_event_, hipEventDefault);
  if (e1 != hipSuccess || e2 != hipSuccess) {
    std::fprintf(stderr,
                 "[HCTR perfetto] failed to create events for phase '%s' (rank %d): "
                 "start=%s end=%s\n",
                 name_.c_str(), rank_, hipGetErrorString(e1), hipGetErrorString(e2));
  }
}

GpuPhase::~GpuPhase() {
  if (start_event_) hipEventDestroy(start_event_);
  if (end_event_) hipEventDestroy(end_event_);
}

void GpuPhase::record_start(hipStream_t stream, const std::string& stream_name) {
  if (stream_name_.empty()) stream_name_ = stream_name;
  hipEventRecord(start_event_, stream);
  recorded_this_iter_ = true;
  // Lazily register this phase with its rank (only on first call).
  // For captured-graph phases this happens once; subsequent replays just
  // re-record the same event without touching the registry.
  auto& rs = PerfettoEmitter::instance().rank_state_[rank_];
  bool found = false;
  for (auto* p : rs.registered_phases) {
    if (p == this) { found = true; break; }
  }
  if (!found) rs.registered_phases.push_back(this);
}

void GpuPhase::record_end(hipStream_t stream) {
  hipEventRecord(end_event_, stream);
}

bool GpuPhase::try_harvest(int iter_idx, hipEvent_t iter_base_event,
                           int64_t iter_start_ns) {
  if (!recorded_this_iter_) return false;
  recorded_this_iter_ = false;

  hipError_t qe = hipEventQuery(end_event_);
  if (qe != hipSuccess) {
    hipEventSynchronize(end_event_);
  }

  // Offset from iter_base (start of iter on compute stream) to phase start.
  // If iter_base_event is null (HCTR_NATIVE_TRACE_BASE off), fall back to
  // start_event_ self-relative timing -- phase ts will be approximate
  // (host-side iter_start_ns is used as iter origin).
  float off_ms = 0.0f;
  if (iter_base_event != nullptr) {
    hipError_t e1 = hipEventElapsedTime(&off_ms, iter_base_event, start_event_);
    if (e1 != hipSuccess) return false;
    if (off_ms < 0.0f) off_ms = 0.0f;
  }

  float dur_ms = 0.0f;
  hipError_t e2 = hipEventElapsedTime(&dur_ms, start_event_, end_event_);
  if (e2 != hipSuccess) return false;
  if (dur_ms < 0.0f) dur_ms = 0.0f;

  PerfettoEvent ev;
  ev.ts_us = (iter_start_ns / 1000) + static_cast<int64_t>(off_ms * 1000.0);
  ev.dur_us = static_cast<double>(dur_ms) * 1000.0;
  ev.rank = rank_;
  ev.stream_name = stream_name_;
  ev.name = name_;
  // Phase 20.9: extract category from NV-style "[category] kernel_name"
  // prefix if present, else fall back to legacy "category/kernel_name"
  // (first segment), else default to "phase".
  if (!name_.empty() && name_[0] == '[') {
    auto close = name_.find(']');
    if (close != std::string::npos) {
      ev.cat = name_.substr(1, close - 1);
    } else {
      ev.cat = "phase";
    }
  } else {
    auto slash = name_.find('/');
    ev.cat = (slash != std::string::npos) ? name_.substr(0, slash) : "phase";
  }
  ev.iter_idx = iter_idx;
  // Phase 20.7: copy static phase metadata into this event (zero-cost
  // beyond the string copy; metadata is set once at register_phase).
  ev.args_extra = args_json_;
  PerfettoEmitter::instance().emit_event(std::move(ev));
  return true;
}

// ---------------------------------------------------------------------------
// PerfettoEmitter
// ---------------------------------------------------------------------------

PerfettoEmitter::PerfettoEmitter() {
  rank_state_.resize(64);  // enough for any realistic node
  events_.reserve(1 << 20);  // reserve 1M events up-front
}

PerfettoEmitter::~PerfettoEmitter() {
  if (flushed_at_exit_.exchange(true)) return;
  if (!native_trace_enabled()) return;
  // Best-effort flush per-rank at shutdown. Pure host-side I/O; no HIP calls.
  for (int r = 0; r < static_cast<int>(rank_state_.size()); ++r) {
    flush(r);
  }
}

PerfettoEmitter& PerfettoEmitter::instance() {
  // Heap-allocated leaked singleton + atexit hook. Avoids Meyers-singleton
  // dtor order issues vs HIP runtime teardown (we only write to host memory
  // in flush(), so atexit ordering is fine).
  static PerfettoEmitter* inst = []() {
    auto* p = new PerfettoEmitter();
    std::atexit([] {
      if (!native_trace_enabled()) return;
      // Use a separate pointer to avoid recursion through instance().
      // We just flush whatever's accumulated.
      PerfettoEmitter& e = PerfettoEmitter::instance();
      if (e.flushed_at_exit_.exchange(true)) return;
      for (int r = 0; r < static_cast<int>(e.rank_state_.size()); ++r) {
        e.flush(r);
      }
    });
    return p;
  }();
  return *inst;
}

GpuPhase* PerfettoEmitter::register_phase(int rank, const std::string& name) {
  if (!native_trace_enabled()) return nullptr;
  std::lock_guard<std::mutex> lk(mu_);
  auto p = std::make_unique<GpuPhase>(rank, name);
  GpuPhase* raw = p.get();
  all_phases_.push_back(std::move(p));
  return raw;
}

GpuPhase* PerfettoEmitter::register_phase(int rank, const std::string& name,
                                          const std::string& args_json) {
  GpuPhase* p = register_phase(rank, name);
  if (p) p->set_args_json(args_json);
  return p;
}

GpuPhase* PerfettoEmitter::register_phase_for_kernel(
    int rank, const std::string& name, const void* func_ptr) {
  // Resolve kernel symbol once at registration. kernel_name_from_ptr()
  // is safe when native tracing is OFF (returns "" without touching HIP)
  // and is internally cached, so registering many phases that share one
  // kernel is O(1) after first lookup.
  std::string kn = kernel_name_from_ptr(func_ptr);
  std::string args_json;
  if (!kn.empty()) {
    // Build a JSON fragment WITHOUT outer braces (the format expected
    // by GpuPhase::args_json_ / PerfettoEmitter::flush()).
    args_json = "\"kernel\":\"" + escape_for_args_json(kn) + "\"";
  }
  return register_phase(rank, name, args_json);
}

void PerfettoEmitter::register_graph_event_chain(
    int rank, const std::string& stream_name, hipEvent_t graph_start_event,
    const std::vector<hipEvent_t>& events, const std::vector<std::string>& names) {
  if (!native_trace_enabled()) return;
  if (rank >= static_cast<int>(rank_state_.size())) return;
  std::lock_guard<std::mutex> lk(mu_);
  GraphEventChain chain;
  chain.rank = rank;
  chain.stream_name = stream_name;
  chain.graph_start_event = graph_start_event;
  chain.events = events;
  chain.names = names;
  rank_state_[rank].graph_chains.push_back(std::move(chain));
}

void PerfettoEmitter::register_graph_clock_chain(
    int rank, const std::string& stream_name,
    unsigned long long* buf_device, unsigned long long* buf_host,
    size_t num_slots, const std::vector<std::string>& names) {
  if (!native_trace_enabled()) return;
  if (rank >= static_cast<int>(rank_state_.size())) return;
  std::lock_guard<std::mutex> lk(mu_);
  GraphClockChain chain;
  chain.rank = rank;
  chain.stream_name = stream_name;
  chain.buf_device = buf_device;
  chain.buf_host = buf_host;
  chain.num_slots = num_slots;
  chain.names = names;
  // Read the wall-clock rate once. hipDeviceAttributeWallClockRate returns
  // KHz (ticks per millisecond). On gfx9x0 this is typically 100000 (100 MHz).
  int rate_khz = 100000;
  hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, rank);
  chain.clock_rate_khz = static_cast<double>(rate_khz);
  rank_state_[rank].graph_clock_chains.push_back(std::move(chain));
}

// Iter range for tracing. Defaults: skip first 2 iters (warmup), then trace
// 5 iters. Override via env:
//   HCTR_NATIVE_TRACE_BEGIN=N -- first iter to record (default 5)
//   HCTR_NATIVE_TRACE_END=N   -- one-past-last iter to record (default 10)
// The legacy HCTR_NATIVE_TRACE_SKIP_WARMUP=N is honored as a back-compat
// alias for BEGIN (and END defaults to BEGIN+5 when only the alias is set).
static int trace_begin_iter() {
  static const int n = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_BEGIN");
    if (e && e[0]) { int v = std::atoi(e); return v >= 0 ? v : 5; }
    e = std::getenv("HCTR_NATIVE_TRACE_SKIP_WARMUP");
    if (e && e[0]) { int v = std::atoi(e); return v >= 0 ? v : 5; }
    return 5;
  }();
  return n;
}
static int trace_end_iter() {
  static const int n = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_END");
    if (e && e[0]) return std::atoi(e);
    return trace_begin_iter() + 5;  // default: 5-iter window
  }();
  return n;
}

void PerfettoEmitter::begin_iter(int rank, int iter_idx, hipStream_t base_stream) {
  if (!native_trace_enabled()) return;
  if (rank >= static_cast<int>(rank_state_.size())) return;
  RankState& rs_clear = rank_state_[rank];
  // Phase 20.9g: ALWAYS clear recorded_this_iter_ flags on every begin_iter,
  // including warmup iters. Otherwise a phase that ran once during warmup
  // (e.g. sp/bucket_range -- only runs when batch size changes, on iter 0)
  // leaves its flag set; when the trace window opens, try_harvest fires on
  // stale start_/end_events from iter 0 and computes a multi-ms elapsed
  // that visually swallows the prefetch lane.
  for (GpuPhase* p : rs_clear.registered_phases) {
    p->mark_unrecorded();
  }
  // Mark this iter as out-of-window by clearing the host-side timer; end_iter
  // will see iter_start_ns==0 and skip harvest. Prevents stale state from
  // a prior in-window iter being re-harvested.
  if (iter_idx < trace_begin_iter() || iter_idx >= trace_end_iter()) {
    rank_state_[rank].iter_start_ns = 0;
    return;
  }
  RankState& rs = rank_state_[rank];
  rs.iter_idx = iter_idx;
  rs.iter_start_ns = now_ns();
  // Env-gated iter-base event: HCTR_NATIVE_TRACE_BASE=1 to enable. Skipping
  // it just means phase ts are relative to host iter_start_ns + their own
  // record_start (less precise sub-iter alignment but still usable).
  static const bool record_base = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_BASE");
    return e != nullptr && e[0] == '1';
  }();
  if (record_base) {
    if (rs.iter_base_event == nullptr) {
      hipEventCreate(&rs.iter_base_event);
    }
    hipEventRecord(rs.iter_base_event, base_stream);
  }
  // NOTE: do NOT mark all registered phases as fresh. Phases that didn't
  // run record_start this iter (e.g., conditional code paths like
  // sp/bucket_range which only runs on batch size change) would otherwise
  // re-emit STALE event timestamps from the last iter they actually ran.
  // record_start already sets recorded_this_iter_ when called.
  //
  // For graph-captured phases (events re-recorded by replay without
  // record_start running), users currently can't get per-iter data via
  // this mechanism -- but our hipStreamIsCapturing gate skips those
  // phases anyway, so this is correctly a no-op for them.
}

void PerfettoEmitter::end_iter(int rank) {
  if (!native_trace_enabled()) return;
  if (rank >= static_cast<int>(rank_state_.size())) return;
  RankState& rs = rank_state_[rank];
  // Skip warmup iters: begin_iter never started the timer, iter_start_ns=0.
  if (rs.iter_start_ns == 0) return;
  // Harvest from ALL registered phases for this rank. Captured-graph phases
  // get their events re-recorded each replay; we harvest every iter via
  // hipEventQuery and append a per-iter PerfettoEvent.
  for (GpuPhase* p : rs.registered_phases) {
    p->try_harvest(rs.iter_idx, rs.iter_base_event, rs.iter_start_ns);
  }
  // Phase 20.8: harvest graph-event chains for this rank. Each event in
  // the chain was recorded as a graph node by the captured graph's replay
  // (zero per-replay host overhead). We compute per-kernel duration as
  // elapsedTime(prev_event, this_event) here on the host (~1us per pair).
  for (const auto& chain : rs.graph_chains) {
    if (chain.events.empty() || chain.graph_start_event == nullptr) continue;
    // Wait for the LAST event in the chain to be ready (bounded by the
    // small tail of GPU work for this iter). Earlier events are guaranteed
    // ready once the last is ready (same stream, monotonic graph order).
    hipEvent_t last_event = chain.events.back();
    if (hipEventQuery(last_event) != hipSuccess) {
      hipEventSynchronize(last_event);
    }
    // Offset from iter_base (host-side iter start) to graph_start_event.
    float graph_start_off_ms = 0.0f;
    if (rs.iter_base_event != nullptr) {
      hipError_t e = hipEventElapsedTime(&graph_start_off_ms, rs.iter_base_event,
                                          chain.graph_start_event);
      if (e != hipSuccess) graph_start_off_ms = 0.0f;
      if (graph_start_off_ms < 0.0f) graph_start_off_ms = 0.0f;
    }
    // Walk the chain emitting one PerfettoEvent per kernel.
    hipEvent_t prev = chain.graph_start_event;
    float cumulative_off_ms = graph_start_off_ms;
    for (size_t i = 0; i < chain.events.size(); ++i) {
      float dur_ms = 0.0f;
      hipError_t e = hipEventElapsedTime(&dur_ms, prev, chain.events[i]);
      if (e != hipSuccess) {
        prev = chain.events[i];
        continue;
      }
      if (dur_ms < 0.0f) dur_ms = 0.0f;
      PerfettoEvent ev;
      ev.ts_us = (rs.iter_start_ns / 1000) +
                 static_cast<int64_t>(cumulative_off_ms * 1000.0);
      ev.dur_us = static_cast<double>(dur_ms) * 1000.0;
      ev.rank = chain.rank;
      ev.stream_name = chain.stream_name;
      ev.name = (i < chain.names.size()) ? chain.names[i] : std::string("g/kernel_?");
      // Build category from leading segment of name.
      auto slash = ev.name.find('/');
      ev.cat = (slash != std::string::npos) ? ev.name.substr(0, slash) : "g";
      ev.iter_idx = rs.iter_idx;
      cumulative_off_ms += dur_ms;
      prev = chain.events[i];
      emit_event(std::move(ev));
    }
  }
  // Phase 20.9: clock-writer chain harvest (in-graph kernel-based timing).
  // After replay completes, hipMemcpy the device clock buffer to host and
  // compute per-kernel durations from tick deltas.
  for (const auto& cchain : rs.graph_clock_chains) {
    if (cchain.num_slots < 2 || cchain.buf_device == nullptr) continue;
    size_t bytes = cchain.num_slots * sizeof(unsigned long long);
    hipError_t e = hipMemcpy(cchain.buf_host, cchain.buf_device, bytes,
                              hipMemcpyDeviceToHost);
    if (e != hipSuccess) continue;
    // First slot is graph start; subsequent N slots are after each kernel.
    unsigned long long t0 = cchain.buf_host[0];
    // Convert tick delta to microseconds: us = ticks / (clock_rate_khz)
    // because clock_rate_khz = ticks per millisecond -> us per tick = 1/khz.
    double us_per_tick = 1000.0 / cchain.clock_rate_khz;
    double cumulative_off_us = 0.0;
    unsigned long long prev_tick = t0;
    for (size_t i = 1; i < cchain.num_slots; ++i) {
      unsigned long long t = cchain.buf_host[i];
      // Guard against unwritten or wrap-around values.
      if (t < prev_tick) { prev_tick = t; continue; }
      double dur_us = static_cast<double>(t - prev_tick) * us_per_tick;
      PerfettoEvent ev;
      ev.ts_us = (rs.iter_start_ns / 1000) +
                  static_cast<int64_t>(cumulative_off_us);
      ev.dur_us = dur_us;
      ev.rank = cchain.rank;
      ev.stream_name = cchain.stream_name;
      ev.name = (i - 1 < cchain.names.size()) ? cchain.names[i - 1]
                                              : std::string("[graph] kernel_?");
      // Category from "[bracket]" prefix
      if (!ev.name.empty() && ev.name[0] == '[') {
        auto close = ev.name.find(']');
        if (close != std::string::npos) ev.cat = ev.name.substr(1, close - 1);
        else ev.cat = "graph";
      } else {
        ev.cat = "graph";
      }
      ev.iter_idx = rs.iter_idx;
      cumulative_off_us += dur_us;
      prev_tick = t;
      emit_event(std::move(ev));
    }
  }
  // Also emit a top-level iter range on a synthetic "iter" lane.
  int64_t iter_end_ns = now_ns();
  PerfettoEvent iter_ev;
  iter_ev.ts_us = rs.iter_start_ns / 1000;
  iter_ev.dur_us = static_cast<double>(iter_end_ns - rs.iter_start_ns) / 1000.0;
  iter_ev.rank = rank;
  iter_ev.stream_name = "iter";
  char nm[64];
  std::snprintf(nm, sizeof(nm), "iter_%d", rs.iter_idx);
  iter_ev.name = nm;
  iter_ev.cat = "iter";
  iter_ev.iter_idx = rs.iter_idx;
  emit_event(std::move(iter_ev));

  // Phase 20.7 fix: only flush ONCE at end of trace window. Periodic
  // mid-trace flush caused multi-rank simultaneous NFS writes (~80ms
  // visible as a host gap between two adjacent iter events). Defer
  // all writes to (a) the LAST traced iter of the BEGIN..END window
  // and (b) process atexit. Trade-off: if process is killed before
  // the last iter, we lose data -- but BEGIN..END is small (~5 iters)
  // so this is acceptable.
  //
  // Legacy HCTR_NATIVE_TRACE_FLUSH_EVERY env var still honored for
  // back-compat (set to a big number to effectively disable).
  static const int flush_every = []() {
    const char* e = std::getenv("HCTR_NATIVE_TRACE_FLUSH_EVERY");
    if (e == nullptr || e[0] == '\0') return 999999;  // off by default
    int v = std::atoi(e);
    return v > 0 ? v : 999999;
  }();
  // Always flush right after the last in-window iter (so trace is
  // available on disk even if process is killed before atexit).
  if (rs.iter_idx == trace_end_iter() - 1) {
    flush(rank);
  } else if (rs.iter_idx > 0 && (rs.iter_idx % flush_every) == 0) {
    flush(rank);
  }
  // Clear timer so subsequent end_iter() calls without a matching begin_iter
  // (e.g., racy double-end, out-of-window iters) don't re-harvest.
  rs.iter_start_ns = 0;
}

void PerfettoEmitter::emit_event(PerfettoEvent e) {
  std::lock_guard<std::mutex> lk(mu_);
  events_.push_back(std::move(e));
}

void PerfettoEmitter::emit_host_event(int rank, const std::string& lane_name,
                                       const std::string& event_name,
                                       int64_t ts_ns, int64_t dur_ns,
                                       const std::string& args_extra) {
  if (!native_trace_enabled()) return;
  // Phase 20.9e: only emit if we're inside the active trace iter window.
  // iter_start_ns is set by begin_iter ONLY for in-window iters (out-of-
  // window begin_iter resets it to 0). Without this check, host timers
  // fire for ALL iters including iters 10..24 which are post-window --
  // creating phantom events that stretch the visible trace span by ~500ms.
  if (rank < 0 || rank >= static_cast<int>(rank_state_.size())) return;
  if (rank_state_[rank].iter_start_ns == 0) return;
  int iter_idx = rank_state_[rank].iter_idx;
  PerfettoEvent ev;
  ev.ts_us = ts_ns / 1000;
  ev.dur_us = static_cast<double>(dur_ns) / 1000.0;
  ev.rank = rank;
  ev.stream_name = lane_name;
  ev.name = event_name;
  // Extract NV-style [bracket] cat or default to "host_api".
  if (!event_name.empty() && event_name[0] == '[') {
    auto close = event_name.find(']');
    if (close != std::string::npos) ev.cat = event_name.substr(1, close - 1);
    else ev.cat = "host_api";
  } else {
    ev.cat = "host_api";
  }
  ev.iter_idx = iter_idx;
  ev.args_extra = args_extra;
  emit_event(std::move(ev));
}

// Phase 20.10 (Path A, 2026-05-20): ScopedGpuPhase impl. When the phase
// is non-null AND HCTR_ROCTX=1, also push a ROCTX range with the phase
// name so the cold-pass rocprofv3 --marker-trace captures every native
// phase boundary. The ScopedRange is heap-allocated to avoid pulling
// rocprofiler-sdk-roctx headers into perfetto_emitter.hpp.
ScopedGpuPhase::ScopedGpuPhase(GpuPhase* phase, hipStream_t stream,
                                 const std::string& stream_name)
    : phase_(phase), stream_(stream), roctx_range_(nullptr) {
  if (phase_) {
    phase_->record_start(stream_, stream_name);
    if (roctx_enabled()) {
      roctx_range_ = new ScopedRange(phase_->name().c_str());
    }
  }
}

ScopedGpuPhase::~ScopedGpuPhase() {
  if (phase_) {
    if (roctx_range_) {
      delete static_cast<ScopedRange*>(roctx_range_);
    }
    phase_->record_end(stream_);
  }
}

ScopedHostTimer::ScopedHostTimer(int rank, std::string lane, std::string name,
                                   std::string args_extra)
    : active_(native_trace_enabled()),
      rank_(rank),
      lane_(std::move(lane)),
      name_(std::move(name)),
      args_extra_(std::move(args_extra)),
      t0_ns_(0) {
  if (active_) t0_ns_ = now_ns();
}

ScopedHostTimer::~ScopedHostTimer() {
  if (!active_) return;
  int64_t t1 = now_ns();
  PerfettoEmitter::instance().emit_host_event(rank_, lane_, name_,
                                                t0_ns_, t1 - t0_ns_,
                                                args_extra_);
}

void PerfettoEmitter::flush(int rank) {
  // Non-destructive: every flush writes the FULL per-rank history to file.
  // This makes the JSON file always reflect the latest snapshot.
  std::vector<PerfettoEvent> snapshot;
  {
    std::lock_guard<std::mutex> lk(mu_);
    snapshot.reserve(events_.size());
    for (const auto& e : events_) {
      if (e.rank == rank) snapshot.push_back(e);
    }
  }
  if (snapshot.empty()) return;

  std::ostringstream path;
  path << trace_dir() << "/hctr_native_trace_rank" << rank << ".json";
  std::ofstream out(path.str());
  if (!out.is_open()) {
    std::fprintf(stderr, "[HCTR perfetto] failed to open %s for writing\n",
                 path.str().c_str());
    return;
  }
  out << "{\n  \"displayTimeUnit\": \"ms\",\n  \"traceEvents\": [\n";
  bool first = true;
  for (const auto& ev : snapshot) {
    if (!first) out << ",\n";
    first = false;
    out << "    {"
        << "\"name\":\"" << json_escape(ev.name) << "\","
        << "\"cat\":\"" << json_escape(ev.cat) << "\","
        << "\"ph\":\"X\","
        << "\"ts\":" << ev.ts_us << ","
        << "\"dur\":" << std::fixed << std::setprecision(3) << ev.dur_us << ","
        << "\"pid\":" << ev.rank << ","
        << "\"tid\":\"" << json_escape(ev.stream_name) << "\""
        << ",\"args\":{\"iter\":" << ev.iter_idx;
    if (!ev.args_extra.empty()) {
      // Static per-phase metadata (PyTorch-profiler-style): GEMM M/N/K,
      // demangled kernel name, grid/block, dtype, etc. Pre-formatted as
      // a JSON fragment without outer braces, inlined after iter.
      out << "," << ev.args_extra;
    }
    out << "}}";
  }
  out << "\n  ]\n}\n";
  out.close();
  std::fprintf(stderr,
               "[HCTR perfetto] wrote %zu events to %s\n",
               snapshot.size(), path.str().c_str());
}

}  // namespace tracing
}  // namespace HugeCTR
