/*
 * Phase 20 (2026-05-16): minimal ROCTX RAII wrapper for HCTR.
 *
 * Goal: tag pipeline phases (fwd / bwd / embedding_* / opt_emb / comm)
 * so that an rocprofv3 trace captured with --marker-trace shows
 * Perfetto host-side ranges around the corresponding hipLaunchKernel
 * calls and GPU kernel executions. rocprofv3 already emits a
 * correlationId on every host API event AND the matching GPU kernel
 * event, and the converter (rocprofv3_to_perfetto_annotated.py) already
 * draws Perfetto flow arrows between them -- so once the ranges are
 * here, we get a PyTorch-profiler-style nested view for free.
 *
 * Default OFF (no overhead) -- set HCTR_ROCTX=1 in the run env to
 * enable. roctx calls are themselves zero-cost when no profiler is
 * attached (the runtime returns immediately), so the env gate is purely
 * a belt-and-suspenders safety knob.
 */
#pragma once

#include <rocprofiler-sdk-roctx/roctx.h>

#include <cstdlib>
#include <utility>

namespace HugeCTR {
namespace tracing {

// Cached env check -- HCTR_ROCTX=1 enables. Default OFF.
inline bool roctx_enabled() {
  static const bool on = []() {
    const char* e = std::getenv("HCTR_ROCTX");
    return e != nullptr && e[0] == '1';
  }();
  return on;
}

// RAII range: push at construction, pop at destruction. Safe to nest.
class ScopedRange {
 public:
  explicit ScopedRange(const char* name)
      : active_(roctx_enabled() && name != nullptr && name[0] != '\0') {
    if (active_) {
      roctxRangePushA(name);
    }
  }
  ~ScopedRange() {
    if (active_) {
      roctxRangePop();
    }
  }
  ScopedRange(const ScopedRange&) = delete;
  ScopedRange& operator=(const ScopedRange&) = delete;
  ScopedRange(ScopedRange&& other) noexcept : active_(other.active_) { other.active_ = false; }

 private:
  bool active_;
};

// Instant marker (no scope) -- useful for iter boundary timestamps.
inline void mark(const char* name) {
  if (roctx_enabled() && name != nullptr) {
    roctxMarkA(name);
  }
}

}  // namespace tracing
}  // namespace HugeCTR
