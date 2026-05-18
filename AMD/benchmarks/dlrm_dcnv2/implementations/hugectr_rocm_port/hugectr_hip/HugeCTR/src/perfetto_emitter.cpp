/*
 * Phase 20.4 (2026-05-17): HCTR-native Perfetto emitter implementation.
 * See perfetto_emitter.hpp for design rationale.
 */
#include <perfetto_emitter.hpp>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

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
  // Timing-enabled events (default flags). The cost is small and we need
  // hipEventElapsedTime() at harvest time.
  hipError_t e1 = hipEventCreate(&start_event_);
  hipError_t e2 = hipEventCreate(&end_event_);
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
  // Build category from leading segment of name ("fwd", "bwd", "comm", ...)
  auto slash = name_.find('/');
  ev.cat = (slash != std::string::npos) ? name_.substr(0, slash) : "phase";
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
