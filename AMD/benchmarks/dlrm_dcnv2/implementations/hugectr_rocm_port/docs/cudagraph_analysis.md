# HIP / CUDA-Graph analysis for HCTR-on-AMD DLRM-DCNv2

Status: 2026-05-17 (third major rewrite — supersedes the 2026-05-16
edition; primary root cause was re-identified after host CPU track
analysis)
Author: chcai (HCTR ROCm port)
Scope: everything we've learned about CUDA-Graph capture / cross-stream
concurrency / host-side dispatch in HugeCTR when ported to AMD MI350X
(gfx950) via HIP, from Phase 14 through Phase 20.

This document is the single reference for anyone trying to close the
AMD↔NV concurrency gap at the CUDA-Graph layer.  The rolling
experimental log is `perf_push_status.log` at the repo root — anything
not summarized here lives there.

---

## 1. TL;DR

- NV HugeCTR's pattern: **capture only the network compute** in one
  small CUDA graph, run all embedding work **eagerly each iter between
  graph launches**, express cross-graph dependencies via
  `cudaEventRecordExternal` / `cudaEventWaitExternal`.  Few hundred LOC
  of scheduling in `pipeline.cpp` + `model_pipeline.cpp` (§4).
- **Our port faithfully reproduces that pattern** (verified line-by-line
  against NVIDIA-Merlin/HugeCTR HEAD; see §7).
- **The dominant AMD-vs-NV bs=1× throughput gap is HOST-SIDE
  `hipGraphLaunch` latency**, NOT cross-stream concurrency, RCCL,
  hipBLASLt, or the External-flag bug.  Measured (§2):
  - NV `cudaGraphLaunch` host call: **0.349–0.751 ms** per iter
  - AMD `hipGraphLaunch` host call: **5.422–5.577 ms** per iter
  - → AMD per-iter spends ~31 % of every iter blocked in the host call
    that's supposed to be async queue submission
- **AMD has explicitly fixed this in ROCm 7.11 / 7.12 / 7.13** (§3) —
  packet batch-dispatch optimization, segment scheduling, pre-computed
  segment dependencies at instantiation.  Our currently shipped
  baseline is **ROCm 7.2.1**, which predates ALL of these fixes.
- **AMD ALSO has a real one-line CLR bug** —
  `Graph::AddExternalEventWaitNode` type confusion (§5.A) — that
  crashes whenever HCTR uses NV's External-wait pattern.  We built a
  HCTR-side public-API workaround (T0''') that works around it.  Verified
  to recover NV-style cross-stream wait semantics, but **no measurable
  perf gain** because cross-stream sync isn't the actual bottleneck.
- The list of things we tried that turned out to be dead ends (full
  story in §8): MEGA-graph fork/join, wide graph capture, loss-memset
  fuse, RCCL stream priority, CU-mask on RCCL streams.  Most reverted.
  The few that landed (External-wait workaround, capture-mode switch,
  Phase-14-era throughput knobs) are documented as kept code.
- Investigation infrastructure shipped in Phase 20: **ROCTX phase
  tagging** end-to-end (§10).  Trace pipeline now emits PyTorch-
  profiler-style nested `iter_N → fwd/* → bwd/* → comm/* → opt/*`
  host-side ranges with correlationId flow arrows to GPU kernels.
  Default ON in the trace script, zero overhead in production.
- **Estimated AMD-vs-NV gap if AMD ships a ROCm with the host-launch
  fix**: ~+15–18 % throughput (14.4 → ~17 M sps), without HCTR changes
  (§11).

---

## 2. The actual root cause: host-side `hipGraphLaunch` latency

### 2.1 What the trace shows

Re-captured `perfetto_lossfuse1` (5 iters in steady state with the
`HCTR_ROCTX=1` / `--marker-trace` overlay).  Looking at the **host CPU
track** rather than GPU streams:

| Metric | NV B200 (rank 0) | AMD MI350X (rank 0) |
|---|---|---|
| `cudaGraphLaunch` / `hipGraphLaunch` count in window | 10 | 6 (5 + warm-up) |
| per-call duration | **0.349 – 0.751 ms** | **5.422 – 5.577 ms** |
| overlap with each other | host calls overlap freely | **strictly sequential** (each ends before next starts) |
| total host time blocked in `*GraphLaunch` | ~3.4 ms over 12 ms window (28 %) | **27.5 ms over 71 ms window (39 %)** |
| host `hipStreamSynchronize` between launches | trivial | 0.8 – 2.2 ms per iter |
| AMD per-node graph-launch overhead (derived) | ~4 µs/node | **~27 µs/node** (≈7× NV) |

NV's `cudaGraphLaunch` is *async queue submission* — the host returns
within < 0.8 ms and is free to start preparing the next iter's data
load and dispatch.  AMD's `hipGraphLaunch` blocks the host for ~5.5 ms
before returning, so the next iter's dispatch can't start.  That's the
shape of the bottleneck.

### 2.2 Per-iter accounting in production

Production iter is ~3.9 ms.  If the trace's host-side blocking
scales 1:1 with the trace-vs-prod wall ratio (88 ms / 19.5 ms ≈ 4.6×),
the production `hipGraphLaunch` host call blocks for ~1.2 ms per iter.
That's ≈ 31 % of every iter spent on the host inside graph launch.

### 2.3 Why every earlier diagnosis missed this

Looking at the GPU streams alone, the bs=1× critical-path wait at the
iter boundary *looks like* `computation_stream` waiting on
`embedding_a2a`'s backward alltoall.  The GPU-side dependency chain
(alltoall_bwd → opt_emb) is real — but it sits BEHIND the host-blocking
`hipGraphLaunch` call, so even if cross-stream sync were instantaneous
the iter wall would still be dominated by host dispatch overhead.

The Phase 20 trace re-analysis caught this only after we added the
`--marker-trace` overlay and looked at host APIs alongside GPU events.
(See §9 for the corresponding lesson.)

### 2.4 Source-level path

Walking `hipGraphLaunch` in the local CLR develop branch tree
(`/home/chcai/rocm_source/rocm-systems/projects/clr/hipamd/src/`):

- `hip_graph.cpp:1648  hipGraphLaunch` →
- `hip_graph.cpp:1635  hipGraphLaunch_common` →
- `hip_graph.cpp:1629  ihipGraphLaunch` →
- `hip_graph_internal.cpp:2318  GraphExec::Run` — the per-launch hot path

`GraphExec::Run` (develop branch) has THREE dispatch paths gated by
`use_segment_scheduling_`:

| Path | When | Behavior |
|---|---|---|
| `EnqueueSegmentedGraph` | `use_segment_scheduling_ && instantiateDeviceId_ == launch_stream->DeviceId()` | New (HIP 7.12+).  Single `AccumulateCommand` per stream batches all kernel dispatches in one barrier; HW events allocated and patched in pre-formed barrier packets. |
| Topo-loop with `CreateCommand` per node | cross-device launch | Legacy |
| `RunNodes()` (legacy per-node walk) | else | Legacy.  Each node calls `CreateCommand` + `EnqueueCommands` individually.  ~27 µs/node host overhead in our measurements. |

**ROCm 7.2.1 (our build) has only the legacy path** — `use_segment_scheduling_`
is never set, so `RunNodes()` is what runs.  That's the path with the
27 µs/node overhead and the strictly-sequential `hipGraphLaunch` calls.

---

## 3. ROCm CHANGELOG evidence — fix exists, ships in 7.11 / 7.12 / 7.13

Direct quotes from `ROCm/rocm-systems/projects/clr/CHANGELOG.md`
(develop branch as of 2026-05-16):

### HIP 7.11 for ROCm 7.11 (Added → Optimized)

> *"Packet batch-dispatch optimization: A new graph-segment scheduling
> mechanism has been added to the HIP runtime to reduce CPU overhead
> during HIP graph launches. It uses hierarchical path discovery to
> construct execution segments that can be dispatched efficiently in
> parallel, replacing the traditional topological-ordering approach."*
>
> *"Improved `hipGraphLaunch` parallelism for complex data-parallel
> graphs. The HIP runtime now eliminates recursion, applies topological
> ordering, and removes an extra loop in `hipGraphLaunch` to streamline
> execution."*

### HIP 7.12 for ROCm 7.12 (Optimized)

> *"HIP Graph Segmented Execution: Graph nodes are grouped into segments
> and dispatched across multiple GPU streams to enable parallel
> execution.  Batching: Each stream receives a single `AccumulateCommand`
> that aggregates all kernel dispatches and submits them efficiently as
> one batch."*
>
> *"Optimized graph stream synchronization by eliminating duplicate
> marker creation when syncing streams back to the launch stream."*
>
> *"Resolved a graph node scheduling issue in multistream execution
> that, in some cases, led to unnecessary kernel-execution stalls."*

### ROCm 7.13 commit 5d1ed85 (2026-04-14)

> *"clr: pre-compute graph segment dependency at instantiate"
>
> "Pre-form barrier packets and generate a patch list at
> hipGraphInstantiate, eliminating per-launch dependency resolution
> and signal acquisition overhead."
>
> "Allocate HW events at hipGraphLaunch, patch them into pre-formed
> barrier/completion packets via ApplyHwEventPatches."
>
> "Single AccumulateCommand for entire launch."*

### HIP 7.2.1 (our shipping baseline)

Only stream-capture validation, `hipEventQuery` / `hipEventSynchronize`
return-value, batch-dispatch doorbell CPU-hang prevention, and
`AMD_DIRECT_DISPATCH` env-var deprecation.  **NONE of the
`hipGraphLaunch` CPU-overhead optimizations.**

### Confirmation in the open-source tracker

- `ROCm/clr#77` "*[Issue]: No performance improvement using hipGraph*"
  — AMD developers acknowledge "hipGraph currently does not provide as
  much advantage as CUDA graphs", improvements focused on MI300.
- The fix tracking PR is `ROCm/rocm-systems#4634` (the April 2026
  commit above).

### What this means for us

If we re-baseline on ROCm 7.11+ AND nothing else regresses, the host
`hipGraphLaunch` time should drop substantially.  Open question (P2):
why did our prior 7.13 / 7.14 nightly A/B show **–11 % throughput
regression** at default config?  Could be hipBLASLt 1.3 TensileLibrary
regression, RCCL build delta, or some other unrelated change.  Needs
isolation.

---

## 4. NV HCTR's canonical concurrency pattern

NVIDIA HugeCTR's stream/event scheduling framework lives in
`HugeCTR/src/pipeline.cpp` and is mirrored 1:1 in our port:

- `StreamContextScheduleable` — a unit of work with optional
  `set_stream("mp"|"dp"|...)`, `wait_event(events, external)`,
  `record_done(external, flags)`, and a `workload` callback.
- `GraphScheduleable` — wraps N scheduleables in a single
  `cudaStreamBeginCapture / EndCapture` window.  Lazy-initialized on
  the first call; subsequent calls just `cudaGraphLaunch`.
- `Pipeline` — sequential list of scheduleables run per training iter.

In `Model::create_train_pipeline_with_ebc`
(`HugeCTR/src/pybind/model_pipeline.cpp`) the construction is:

```text
Pipeline scheduleable_list = [
    distribute_data,         // default
    ebc_mp_model_forward,    // mp stream
    ebc_mp_network_forward,  // mp stream
    ebc_dp_forward,          // dp stream
    network_graph,           // captured -- contains the 7 network ops
    ebc_mp_backward_index_calculation,  // dp stream
    ebc_dp_backward_index_calculation,  // dp stream
    ebc_mp_network_backward, // mp stream
    ebc_dp_local_reduce,     // dp stream
    network_exchange_wgrad,  // dedicated RCCL stream
    ebc_dp_allreduce,        // dedicated RCCL stream
    update_params,           // default
    ebc_mp_local_reduce,     // mp stream
    ebc_mp_update,           // mp stream
    ebc_dp_update,           // dp stream
    sync_back,               // default
]
network_graph = GraphScheduleable(
    network_init, bottom_network_fprop, top_network_fprop,
    init_wgrad, cal_loss, top_network_bprop, bottom_network_bprop)
```

The 7 network compute ops are captured ONCE into a CUDA graph.  Each
training iter runs the outer list in order; when it reaches
`network_graph` it does `cudaGraphLaunch` of the cached graph.

### How NV gets concurrency

1. **All embedding work runs eagerly each iter on its own stream.**  No
   fork/join into the captured graph.  Each eager call returns
   immediately after enqueuing.
2. **Cross-graph dependencies use External-flag events.**  Three
   specific sites in `model_pipeline.cpp` set `use_graph=true` on
   `wait_event` / `record_done`:
   - `bottom_network_fprop->wait_event({done_mp_model_forward}, use_graph)`
   - `top_network_fprop->wait_event({done_dp_forward, done_mp_network_forward}, use_graph)`
   - `top_network_bprop->record_done(use_graph)` → awaited by
     `ebc_mp_network_backward` and `ebc_dp_local_reduce`
3. With `use_graph=true` inside capture,
   `cudaStreamWaitEvent(stream, event, cudaEventWaitExternal)` records
   a `GraphEventWaitNode`.  At replay, that node tells the runtime to
   wait on whatever `done_mp_model_forward` was MOST RECENTLY recorded
   — the EAGER mp forward of the CURRENT iter.  So mp/dp forward run
   in parallel with `distribute_data`; `bottom_network_fprop` (inside
   capture) blocks at replay only until mp forward finishes.

NV explicitly does NOT capture embeddings + network into one MEGA
graph.  Cross-stream capture in multi-stream graphs is fragile even on
CUDA.

---

## 5. Bugs discovered in the HIP runtime

### 5.A — `Graph::AddExternalEventWaitNode` type confusion (THE main bug)

**Location:** `ROCm/rocm-systems/projects/clr/hipamd/src/hip_graph_internal.hpp`
**Versions affected:** all shipping ROCm 7.0 → 7.2.1 → 7.12.0rc1 → 7.13
nightly → 7.14 nightly + current develop head.

The buggy helper:

```cpp
GraphNode* AddExternalEventWaitNode(hip::GraphNode* pDependencies,  // ← WRONG
                                    size_t numDependencies,
                                    hipEvent_t event) {
  GraphNode* node = new GraphEventWaitNode(event);
  for (size_t i = 0; i < numDependencies; i++) {
    pDependencies[i].AddEdgeDep(node);  // ← dot, indexes by sizeof(GraphNode)
  }
  AddNode(node);
  return node;
}
```

Caller in `hip_stream.cpp`:

```cpp
auto lastCapturedNodes = waitStream->GetLastCapturedNodes();
hip::GraphNode* pGraphNode = waitStream->GetCaptureGraph()->AddExternalEventWaitNode(
    reinterpret_cast<hip::GraphNode*>(lastCapturedNodes.data()),  // GraphNode** → GraphNode*
    lastCapturedNodes.size(), event);
```

`lastCapturedNodes` is `std::vector<hip::GraphNode*>` so `.data()` is
`hip::GraphNode**`.  The `reinterpret_cast` tells the compiler it's
`hip::GraphNode*`.  Inside the loop, `pDependencies[i]` then indexes
by `sizeof(hip::GraphNode)` (100+ bytes) instead of
`sizeof(hip::GraphNode*)` (8 bytes) and dereferences garbage memory.

#### Crash signatures across ROCm versions

| Version | Failure mode |
|---|---|
| 7.2.1 | uncaught `std::bad_alloc` |
| 7.2.3 | uncaught `std::bad_alloc` |
| 7.12.0rc1 | `hipErrorOutOfMemory` (caught) |
| 7.13 nightly + Relaxed | `hipErrorOutOfMemory` (caught) |
| 7.13 nightly + ThreadLocal | **segfault** inside `libamdhip64.so.7+0x285e2b` (deepest stack, clearest signature for bug filing) |

#### Contrast — same source file has the correct pattern

In every other graph-add helper (`ihipGraphAddNode`,
`ihipGraphAddKernelNode`, etc.):

```cpp
inline hipError_t ihipGraphAddNode(hip::GraphNode* graphNode, hip::Graph* graph,
                                   hip::GraphNode* const* pDependencies,  // ← CORRECT
                                   size_t numDependencies, ...) {
  ...
  pDependencies[i]->AddEdgeDep(graphNode);  // ← arrow, indexes by sizeof(GraphNode*)
}
```

`AddExternalEventWaitNode` is plainly a sloppy copy-paste of that
pattern that lost the `* const*` and the `->`.

#### Suggested one-line fix

```diff
- GraphNode* AddExternalEventWaitNode(hip::GraphNode* pDependencies, ...)
+ GraphNode* AddExternalEventWaitNode(hip::GraphNode* const* pDependencies, ...)
-     pDependencies[i].AddEdgeDep(node);
+     pDependencies[i]->AddEdgeDep(node);
  // and drop the reinterpret_cast in hip_stream.cpp
```

### 5.B — std::bad_alloc with hipEventWaitExternal + hipEventDisableTiming on 7.2.1

When HCTR creates the cross-graph event with `hipEventDisableTiming`
(NV's default) AND uses it on the External-wait path, the runtime
throws `std::bad_alloc`.  Likely the runtime tries to allocate timer
state proportional to graph nodes for an event flagged as
external-referenced, and the DisableTiming bit interacts badly.

**Hypothesis-2 workaround** (Phase 14s, in `pipeline.cpp:75-80`):
when `HCTR_RESTORE_WAIT_EXTERNAL=1`, force the event creation flag to
`hipEventDefault` instead of `hipEventDisableTiming`.  Verified to
sidestep the bad_alloc on 7.2.1 — but the underlying type-confusion
bug (5.A) still fires elsewhere and crashes the run, so 5.B alone is
not enough.

### 5.C — Subtler CLR behaviors hit while attempting MEGA-graph

Not currently blocking anything (we abandoned MEGA-graph as
architecturally wrong, §8.B) but documented for completeness:

1. **`hipEventRecord` with DEFAULT flags inside capture creates NO
   graph node.**  It only sets `nodesPrevToRecorded_` on the event.
   The `hipEventRecordExternal` flag is what actually creates a
   `GraphEventRecordNode`.  This is a real CUDA-CLR behavior
   difference that's easy to trip on.
2. **`hipStreamWaitEvent`'s IsEventCaptured branch reads
   `e->GetNodesPrevToRecorded()`** as the cross-deps.  But
   `SetNodesPrevToRecorded(lastCapturedNodes)` runs BEFORE the
   External record node is created (see `hip_event.cpp:436-440` and
   `:443+` ordering).  So the wait sees the OLD predecessors, not the
   new External record node itself.
3. **`hipStreamWaitEvent` inside capture does not create a graph edge
   immediately.**  It only appends to `lastCapturedNodes_`, and the
   edge materializes when the NEXT captured op consumes those deps.
   If no subsequent op exists, the fork stream's last node stays as a
   leaf and `hipStreamEndCapture` errors "capturing stream has
   unjoined work" at `hip_graph.cpp:1263`.

We tried 4 successive patches to a fork/join mechanism that would
join mp/dp streams into a captured graph and join them back at the
end.  None worked, even with all 3 issues above worked around.  The
public-API path (`hipGraphAddEventWaitNode` +
`hipStreamUpdateCaptureDependencies`) also failed to produce a
fully-joined graph.  AMD's multi-stream capture machinery appears to
have deeper bugs we haven't exhaustively traced.

### 5.D — Tiny `hipMemsetAsync` graph-replay overhead (~1.5 ms gap)

`hipMemsetAsync(loss, 0, sizeof(float), stream)` followed immediately
by `BinaryCrossEntropy_Kernel<<<…>>>` on the SAME stream produces a
1.5 ms gap in graph replay.  Trace evidence is unambiguous (see
`perfetto_lossfuse1`, `loss.cu:274`).

- HIP 7.12 CHANGELOG mentions a "back memory set (memset) optimization
  to improve how `memset` nodes are processed during graph execution"
  — almost certainly the fix.
- Our HCTR_LOSS_FUSE_ZERO patch (env-gated kernel-zero substitute)
  closes the gap cleanly at the trace level (1.5 ms → 0 µs) but the
  gap is NOT on the critical path at bs=1× — it sits inside the host
  hipGraphLaunch shadow (§2), so removing it doesn't change wall time.
  Patch was reverted in Phase 20 cleanup (§8); see §9 for the lesson.

### 5.E — hipBLASLt back-to-back GEMM dispatch latency in graph replay

Between consecutive `mlp_bwd_dgrad` GEMMs on the same stream we
observed 0.3–0.6 ms gaps with no work on any stream — 17 such gaps
per iter on `computation_stream` (~5 ms in trace, ~1 ms in
production).

- Same general class of issue as 5.D — per-call dispatch overhead
  baked into the captured graph and persisting across replays.
- Likely fixed by the 7.11/7.12 segmented-execution work (§3).
- Not addressable at the HCTR layer beyond reducing GEMM count
  (CK-Tile fused MLP, see §11).

---

## 6. The throughput gap — corrected attribution

### 6.1 Baseline numbers

| Build | Throughput | vs NV B200 |
|---|---|---|
| **NV B200** (CUDA, External-flag wait functional) | 26.33 M sps | 100 % |
| **AMD MI350X production** (External-flag default-OFF because crash) | 14.40 M sps (4-trial avg, σ ≈ 0.06) | 54.7 % |
| AMD MI350X with T0''' workaround (`HCTR_EXTERNAL_WAIT_WORKAROUND=1` + `HCTR_RESTORE_WAIT_EXTERNAL=1`) | 14.21 M sps (2-trial avg) | 54.0 % |

The T0''' workaround is within noise of the baseline (−0.9 %).  Even
with the External-flag path functional, AMD doesn't close the gap.

### 6.2 Where the 46 % gap actually lives (Phase 20 corrected)

| Cause | Layer | Estimated share of gap | Status |
|---|---|---|---|
| **HOST `hipGraphLaunch` latency** (§2) — 5.5 ms / iter trace, ~1.2 ms / iter prod, 10× NV | Vendor (CLR `RunNodes` legacy path) | **~15–18 %** | Fixed in HIP 7.11/7.12/7.13 (§3); not in our 7.2.1 |
| hipBLASLt per-call dispatch latency in graph replay (5.E) | Vendor (CLR + hipBLASLt) | ~5–8 % | Same family as above; same upstream fix |
| Tiny `hipMemsetAsync` graph-replay latency (5.D) | Vendor (CLR) | ~3–5 % (in trace, smaller in prod) | Fixed by HIP 7.12 memset-node optimization |
| RCCL alltoall per-call cost | Vendor (RCCL) | ~5–7 % | NVLS-equivalent for xGMI not in ROCm 7.2.x |
| hipBLASLt MLP kernel exec speed | Vendor (hipBLASLt + TensileLibrary) | ~5–8 % | Fewer auto-tuned gfx950 entries than CUDA has for B200 |
| Misc launch-latency / kernel-speed deltas | Both | residual ~3–5 % | |
| **CUDA-graph external-flag wait area** | HCTR + CLR | **~0 %** | Workaround works (T0''') but no perf gain |

#### Lessons about earlier incorrect attributions

- The "**RCCL `ncclDev` alltoall is 7× slower (~25 % of gap)**"
  conclusion from the 2026-05-16 second pass was **partly trace
  artifact**.  Kernel exec times in the trace are GPU-clock accurate,
  but per-iter wall is inflated ~4.6× by rocprofv3 instrumentation
  overhead.  Once you account for that, RCCL alltoall is meaningfully
  slower than NV but not 7×, and it's no longer on the iter critical
  path because compute_stream is also waiting on host-side launch.
- The "compute_stream has 53 unfilled gaps totaling 19 ms of GPU-wide
  idle" observation is real but DOWNSTREAM of the host-side stall.
  When `hipGraphLaunch` blocks the host, no stream gets new work, so
  the GPU sits fully idle — the unfilled gaps are the visible shadow
  of the host blocking.

### 6.3 Critical-path picture

- **NV** (2.5 ms / iter): `cudaGraphLaunch` returns in 0.6 ms; embedding
  / RCCL / compute overlap cleanly inside the next iter window;
  computation_stream is busy ~75 % of the iter.
- **AMD** (3.9 ms / iter production): `hipGraphLaunch` blocks the host
  for ~1.2 ms; iter dispatch is serialized on the host; the GPU-side
  dependency chain (alltoall_bwd → opt_emb) becomes the *secondary*
  critical path that's only visible after the host bottleneck is
  removed.

---

## 7. Source-level comparison: HCTR port vs NV upstream

Pulled NVIDIA-Merlin/HugeCTR HEAD on 2026-05-16 and diff'd the relevant
files.

| File | Diff status | Notes |
|---|---|---|
| `HugeCTR/src/pipeline.cpp` | functionally identical | `HCTR_RESTORE_WAIT_EXTERNAL` env-gate is an A/B safety switch; enabling it produces the same code path as NV. |
| `HugeCTR/include/pipeline.hpp` | identical + Phase 20 `debug_name_` field for ROCTX | |
| `HugeCTR/src/graph_wrapper.cpp` | NV: `cudaStreamCaptureModeThreadLocal`; ours: `hipStreamCaptureModeRelaxed` by default | HIP 7.0+ docs say only Relaxed is fully supported; ThreadLocal hits open ROCm/hip#3876 + pytorch#177309.  Env-gated via `HCTR_CAPTURE_MODE_RELAXED` (default ON). |
| `HugeCTR/src/gpu_resource.cpp` | NV: `computation_stream_2_` created with default flags (blocking); ours: `hipStreamNonBlocking` | Conscious port fix for HIP behavior; doesn't touch External-flag path. |
| `HugeCTR/include/stream_event_manager.hpp` | identical + optional CU-mask (env-gated, default OFF) | |
| `HugeCTR/src/data_readers/multi_hot/async_data_reader.cpp` | identical port — same `split_schedule_events_` / `d2d_schedule_events_` flow, same Record/Wait External-flag pattern | |
| `HugeCTR/src/pybind/model_pipeline.cpp` | identical scheduling structure — same `wait_event(use_graph)` / `record_done(use_graph)` placement + Phase 20 `set_debug_name()` calls | |

**Verdict: the port is correct.**  Nothing missed in hipification could
cause our concurrency/graph-capture crashes.

NV's actual MLPerf B200 config (`config_b200_1x8.sh` +
`config_common.sh`) has only 3 env vars: `NCCL_NVLS_ENABLE=1`,
`NCCL_GRAPH_REGISTER=0`, `NCCL_LOCAL_REGISTER=0`.  No CUDA-graph-
specific knobs.  They simply trust that the runtime works — and on
CUDA, it does.

---

## 8. Workarounds + things we tried (the complete history)

This section is the catalog of everything that landed, was attempted,
or was reverted, with the reason for each.  Numbered T0..T20 by
chronological phase.

### 8.A — Kept in the tree today

| ID | Phase | What | Default | Reason kept |
|---|---|---|---|---|
| **T0'** | 14p | `hipStreamCaptureModeRelaxed` instead of ThreadLocal in `graph_wrapper.cpp` (gated via `HCTR_CAPTURE_MODE_RELAXED`) | **ON** | Safety: ThreadLocal has multiple open ROCm bugs. |
| **T0''** | 14s | `hipStreamIsCapturing` guard before passing `hipEventWaitExternal` in `pipeline.cpp` and `async_data_reader.cpp` | always | Defensive: falls back to flag=0 outside capture; avoids 5.A crash in non-capture paths. |
| **T0'''** | 14n.7 | Public-API substitute for `Graph::AddExternalEventWaitNode` (gated via `HCTR_EXTERNAL_WAIT_WORKAROUND`, requires `HCTR_RESTORE_WAIT_EXTERNAL=1` to also be enabled) | **OFF** | First-ever successful EXT=1 run on AMD MI350X.  Recovers NV-style wait semantics but no perf gain at bs=1× (see §6); kept for future workloads / shapes / ROCm versions. |
| **Phase 14s record-event flag fix** | 14s | `hipEventDefault` instead of `hipEventDisableTiming` for events used with External wait (gated via `HCTR_RESTORE_WAIT_EXTERNAL`) | **OFF** | Workaround for 5.B std::bad_alloc.  Coupled to T0'''. |
| **Phase 15** | 15 | Dedicated RCCL streams (`set_absolute_stream("rccl_emb_ar")`, `set_absolute_stream("rccl_mlp_wgrad")` in `model_pipeline.cpp`); gated via `HCTR_DEDICATED_RCCL_STREAM` | **ON** | +3.0 % at bs=1× when measured; the AMD-side equivalent of NV cuBLASLt-internal worker streams that hipBLASLt doesn't have. |
| **Phase 16** | 16 | MultiCross dx-memset elimination: first-iter overwrite variant of `vector_mul_fma3_align<__half, 8, 4>` in `multi_cross_layer.cu`; gated via `HCTR_SKIP_DX_MEMSET` | **ON** | +0.7 %; closes the largest single contributor to AMD's `__amd_rocclr_fillBufferAligned` time. |
| **Phase 18** | 18 | `HCTR_ASYNC_WGRAD=0` in `run_apple_to_apple.sh` (NV default is True; AMD-specific config inversion because hipBLASLt has no internal worker streams) | **0** (in run script) | +1.9 % at bs=1×. |
| **Phase 20 ROCTX instrumentation** | 20 | `hctr_tracing.hpp` + `set_debug_name()` on every scheduleable + `iter_N` range in `Model::train`; CMakeLists links `librocprofiler-sdk-roctx`; trace pipeline adds `--marker-trace` + `HCTR_ROCTX=1`. See §10. | OFF in production; **ON** in trace script | Pure instrumentation; zero overhead in production. |

### 8.B — Tried and abandoned (NOT in the tree)

| ID | Phase | What | Outcome | Why not |
|---|---|---|---|---|
| MEGA-graph | 14n.5 + 14n.6 | `GraphScheduleable::set_fork_streams({"mp","dp"})` — wrap embedding fwd + network compute in ONE graph by forking mp/dp streams into the capture and joining back at end. 4 successive patches: v1 (plain hipEventRecord + hipStreamWaitEvent), v2 (extra hipEventRecord after join), v3 (External flags throughout), v4 (explicit public-API joins with `hipStreamGetCaptureInfo_v2` + `hipStreamUpdateCaptureDependencies`) | All 4 crashed with "unjoined work" at EndCapture, or "operation not permitted when stream is capturing" | (a) NV upstream does NOT do this — verified by reading their pipeline.cpp.  (b) AMD's multi-stream capture has the §5.C runtime bugs.  Architecturally wrong dead end. |
| WIDE_GRAPH_CAPTURE | 14n.7 / A4b | Hoist the 3 External-flag cross-graph waits UP to the `network_graph` level so they execute OUTSIDE capture (flag=0) instead of needing External | Worked functionally but mild over-serialization; no perf gain | Orphaned without MEGA_GRAPH. |
| HCTR_LOSS_FUSE_ZERO | 20 | Replace `hipMemsetAsync(loss, 0, sizeof(float))` with 1-thread `zero_one_float_kernel<<<1,1>>>` in `loss.cu`'s BCE path | Closed the 1.5 ms graph-replay gap cleanly at trace level (verified via `perfetto_lossfuse1` side-by-side). Zero wall-time win (+0.32 % within noise) | Gap was NOT on the critical path at bs=1× — host hipGraphLaunch shadow consumes it.  Revert during Phase 20 cleanup. |
| Cleanup attempt: remove External-flag plumbing | 20 | Drop `wait_external_` / `record_external_` from `StreamContextScheduleable` API, drop the External path in `async_data_reader.cpp::schedule_*_here` | **BROKE convergence at iter ~1000** | The `hipEventRecordExternal` flag on `split_schedule_events_` / `d2d_schedule_events_` was load-bearing for cross-iter event semantics, not just a no-op.  Reverted everything to HEAD; re-applied only ROCTX. |

### 8.C — Throughput knob sweep (none landed; see `perf_push_status.log § Comprehensive env-knob sweep`)

| Knob | Effect | Why not landed |
|---|---|---|
| `HCTR_INNERPRODUCT_ASYNC_WGRAD=1` | Neutral | hipBLASLt has no internal worker streams so async-wgrad doesn't overlap anything real |
| `NCCL_BUFFSIZE=32 MB` | −1.1 % | Smaller default (8 MB) wins on our shapes |
| CU-mask on RCCL streams | −24 % at every mask value | RCCL doesn't benefit from CU partitioning on gfx950 |
| `HCTR_RCCL_STREAM_PRIORITY=-1` | Initial 2-trial +1.8 %, but 3-trial validation showed **intermittent first-iter divergence** (1 of 3 runs hit "Loss cannot converge" before iter 200) | HIP runtime scheduling-order non-determinism interacting with FP16 scaler=16348.  Opt-in via env var only. |

---

## 9. Lessons learned and trace-analysis gotchas

Catalogued here in case the same mistakes are tempting again.

1. **`rocprofv3 --kernel-trace --hip-trace --memory-copy-trace
   --rccl-trace` inflates the per-iter wall by ~4–5× on AMD.**  GPU
   kernel exec timestamps are GPU-clock accurate, but the gaps
   between kernels include host instrumentation overhead.  Per-iter
   wall in our trace was 17.65 ms vs production 3.9 ms.  Always
   normalize before quoting "AMD is N× slower than NV."
2. **Aggregate per-stream busy time is NOT the same as actual stream
   concurrency.**  Two streams can each be 90 %-busy and still be
   strictly serial.  Use **pairwise concurrent active time** (time
   when BOTH streams are simultaneously active) instead.  Phase 14r
   "concurrency improved 27 %" was wrong; pairwise re-measurement
   showed 0.30 ms × 0.1 ms × 0.1 ms — essentially serial.
3. **GPU-stream-only analysis cannot see host-side blocking.**  Always
   capture the host CPU track (`--hip-trace` already does this; the
   converter renders it as `host_thread` lanes).  The host
   `hipGraphLaunch` call ALONE explained 31 % of our iter time; we
   missed it for 6 weeks because we only looked at GPU streams.
4. **Add ROCTX phase tagging before looking at unfamiliar traces.**
   Without `iter_N → fwd → bwd → comm → opt` ranges, pinpointing
   which HCTR phase produces a specific GPU-stream gap is detective
   work.  See §10.
5. **Cross-verify a root-cause hypothesis against NV's equivalent
   trace before betting on a fix.**  If NV has the same pattern but
   doesn't suffer the same gap, the gap is downstream of something
   else.  Several Phase 20 dead ends would have been caught earlier
   by this discipline.
6. **Closing a gap at the trace level doesn't mean closing it in
   production.**  HCTR_LOSS_FUSE_ZERO closed a 1.5 ms graph-replay
   gap exactly as designed, but production iter wall didn't move
   because the gap sat behind a bigger host-side block.  Trace-level
   verification proves a patch is *mechanically correct*, not that
   it's *on the critical path*.
7. **Read the CHANGELOG.**  HIP 7.11 / 7.12 / 7.13 ship exactly the
   per-launch CPU overhead reductions we need.  We discovered this
   only after 4 weeks of investigation that could have been
   short-circuited by 30 minutes of CHANGELOG.md reading.
8. **Don't drop API params just because the env-gated path that uses
   them is currently OFF.**  The data-reader External-flag plumbing
   was load-bearing for cross-iter event semantics even with
   `HCTR_RESTORE_WAIT_EXTERNAL=0`; "removing dead code" broke
   convergence at iter 1000.  Surgical revert (HEAD + only ROCTX
   re-applied) was the right recovery.

---

## 10. ROCTX phase-tagging infrastructure (Phase 20)

End-to-end PyTorch-profiler-style host-side phase ranges with
correlationId flow arrows to GPU kernels.

### 10.1 Files

| File | Status | What |
|---|---|---|
| `HugeCTR/include/hctr_tracing.hpp` | **NEW** | RAII `ScopedRange` wrapper around `roctxRangePushA`/`Pop`; env-gated by `HCTR_ROCTX=1` |
| `HugeCTR/include/pipeline.hpp` | +18 LOC | `debug_name_` field + `set_debug_name()` on `StreamContextScheduleable` and `GraphScheduleable` |
| `HugeCTR/src/pipeline.cpp` | +11 LOC | Wraps `run()` body with `ScopedRange(debug_name_)` so every phase auto-tags |
| `HugeCTR/src/pybind/model_pipeline.cpp` | +25 LOC | `set_debug_name()` on 16 scheduleables: `fwd/bmlp`, `fwd/tmlp`, `fwd/ebc_mp_model`, `fwd/ebc_mp_network`, `fwd/ebc_dp`, `fwd/loss`, `bwd/tmlp`, `bwd/bmlp`, `bwd/ebc_mp_network`, `bwd/ebc_mp_idx_calc`, `bwd/ebc_mp_local_reduce`, `bwd/ebc_dp_idx_calc`, `bwd/ebc_dp_local_reduce`, `bwd/init_wgrad`, `comm/mlp_wgrad_allreduce`, `comm/ebc_dp_allreduce`, `opt/mlp_update`, `opt/ebc_mp_update`, `opt/ebc_dp_update`, `data/distribute`, `data/ebc_cache_ddl`, `data/copy_next_iter`, `sync_back`, `graph_network` |
| `HugeCTR/src/pybind/model.cpp` | +13 LOC | `iter_N` range around `Model::train()` body with atomic counter |
| `HugeCTR/src/CMakeLists.txt` | +21 LOC | Auto-detect + link `librocprofiler-sdk-roctx.so.1` |
| `scripts/_trace_and_convert_inner.sh` | +8 LOC | `--marker-trace` flag + `export HCTR_ROCTX=1` |
| `scripts/trace_and_convert.sh` | +1 LOC | Forwards `HCTR_ROCTX` through docker `-e` |
| `scripts/rocprofv3_to_perfetto_annotated.py` | +90 LOC | Ingest `_marker_api_trace.csv`, render ranges on `roctx_ranges` lane with prefix-derived colors |

### 10.2 What the trace looks like now

In Perfetto, host lane shows nested ranges:

```
iter_15 (18.8 ms trace, 4 ms prod)
  ├─ data/ebc_cache_ddl   (1.1 ms)
  ├─ fwd/ebc_mp_model     (0.5 ms)
  ├─ fwd/ebc_mp_network   (0.6 ms)
  ├─ fwd/ebc_dp           (0.9 ms)
  ├─ graph_network        (4.5 ms) ← the captured CUDA graph; this is where hipGraphLaunch blocks
  ├─ bwd/* ...
  ├─ comm/mlp_wgrad_allreduce
  ├─ opt/* ...
  └─ sync_back
```

Per-GPU JSON now has the host phase ranges plus existing flow arrows
from `hipLaunchKernel` to each GPU kernel via correlationId.  Click
any slow GPU kernel and Perfetto walks the arrow back to its host
call AND to the enclosing phase range.

### 10.3 Validation

`perfetto_roctx_phase20` (jid 4975, iters 15..20):

- rocprofv3 emitted `iter15_20_marker_api_trace.csv` with 3 681 ROCTX
  events across 8 GPUs.
- Converter classified 733 ranges per GPU into the steady window:
  203 bwd, 120 data, 120 fwd, 120 opt, 84 graph_network, 80 comm, 6
  iter.
- Each `iter_N` range cleanly contains its sub-phases.

### 10.4 How to use

```bash
# Capture a 5-iter steady-state trace with ROCTX overlay
bash scripts/trace_and_convert.sh <jobid> 15 20 <tag>
# Outputs: results/perfetto_<tag>/iter_steady_gpu{0..7}.json
#          results/perfetto_<tag>/iter_steady_all8gpus.json
```

Default-ON inside the trace script.  To capture *without* ROCTX
(legacy A/B), set `HCTR_ROCTX=0` in the env before invoking.

---

## 11. Follow-up priorities

Ordered by effort × upside.  See `perf_push_status.log` for the rolling
status of each.

### P1 (cheap; do first) — ROCm 7.13/7.14 A/B with default config

The trace evidence + CHANGELOG (§2 + §3) strongly suggests the host
`hipGraphLaunch` bottleneck is fixed in HIP 7.11+.  We saw −11 % at
default config on a prior 7.13/7.14 nightly A/B — that needs
re-isolation.  Plausible candidates: hipBLASLt 1.3 TensileLibrary
regression, RCCL build delta, some other unrelated change.  Steps:

1. Capture a fresh trace on 7.13/7.14 with HCTR_ROCTX=1 (now that
   instrumentation is in place).
2. Compare per-phase wall (`graph_network`, `comm/*`, `opt/*`) vs
   7.2.1 with the same overlay.
3. If `graph_network` drops as expected but another phase regresses,
   that's the culprit to bisect.

Expected upside if `hipGraphLaunch` drops to NV's level: **+15–18 %
throughput** (14.4 → ~17 M sps) WITHOUT any HCTR changes.

### P2 — Reduce graph node count per iter (HCTR-side)

Even on 7.2.1, fewer graph nodes = less per-launch CPU time
(currently ~27 µs/node × ~200 nodes = ~5.5 ms/iter).  Levers:

1. Enable `HCTR_USE_CK_TILE_MLP=1` to fuse multi-GEMM MLPs into single
   CK-Tile kernel launches (code already exists in
   `cktile_mlp_kernel.cu`, partially used).
2. Audit `emb_reduce` / `sparse_prep` for fusion opportunities.

Estimated: each 10 % node-count reduction → ~0.5 ms/iter saved on
7.2.1.  Less impactful on 7.11+.

### P3 — File vendor bugs

1. **`Graph::AddExternalEventWaitNode` type confusion** (5.A) — one
   line fix, with attached minimal repro and patch diff.  Most
   actionable single ask: lets us retire T0''' workaround and run
   with `HCTR_RESTORE_WAIT_EXTERNAL=1` as default.
2. **`hipGraphLaunch` host-side latency vs `cudaGraphLaunch`** (§2)
   — even if 7.11+ already addresses it, file to confirm fix lands
   in 7.2.x LTS.

### P4 — Long-tail vendor asks

1. **hipBLASLt internal worker streams** for GEMM parallelism — the
   largest remaining single win (cuBLASLt-equivalent).  Vendor-deep,
   no public ETA.
2. **NVLS-equivalent for xGMI** in RCCL.  Hardware/runtime; ROCm 7.3+.
   Even larger potential (~8–10 % of throughput).
3. **hipBLASLt TensileLibrary expansion** for gfx950 — more auto-tuned
   kernels for our exact MLP shapes.

---

## 12. Appendix — file paths quick reference

### Our port (HCTR-on-ROCm)

- Graph capture wrapper: `hugectr_hip/HugeCTR/src/graph_wrapper.cpp`
- Stream/event scheduling: `hugectr_hip/HugeCTR/src/pipeline.cpp` +
  `hugectr_hip/HugeCTR/include/pipeline.hpp`
- GPU resource + stream manager: `hugectr_hip/HugeCTR/src/gpu_resource.cpp` +
  `hugectr_hip/HugeCTR/include/gpu_resource.hpp` +
  `hugectr_hip/HugeCTR/include/stream_event_manager.hpp`
- Data reader cross-stream sync:
  `hugectr_hip/HugeCTR/src/data_readers/multi_hot/async_data_reader.cpp`
- DLRM pipeline assembly:
  `hugectr_hip/HugeCTR/src/pybind/model_pipeline.cpp`
- ROCTX tracing helper (Phase 20):
  `hugectr_hip/HugeCTR/include/hctr_tracing.hpp`
- Iter-loop ROCTX entry point:
  `hugectr_hip/HugeCTR/src/pybind/model.cpp` (`Model::train()`)

### Trace pipeline (Phase 20)

- Outer driver: `scripts/trace_and_convert.sh`
- Container payload: `scripts/_trace_and_convert_inner.sh`
- Per-GPU converter: `scripts/rocprofv3_to_perfetto_annotated.py`
- GEMM problem-shape extractor: `scripts/extract_gemm_shapes.py`
- 8-GPU merger: `scripts/merge_perfetto_gpus.py`

### NV upstream (HugeCTR HEAD)

All same paths under https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/HugeCTR

### ROCm CLR (where the runtime bugs live)

- Buggy helper (5.A): `projects/clr/hipamd/src/hip_graph_internal.hpp`
  `Graph::AddExternalEventWaitNode()`
- Caller: `projects/clr/hipamd/src/hip_stream.cpp`
  `hipStreamWaitEvent_common()` External branch
- `hipGraphLaunch` host hot path: `projects/clr/hipamd/src/hip_graph.cpp:1648`
  → `hip_graph_internal.cpp:2318 GraphExec::Run`
- EndCapture join check: `projects/clr/hipamd/src/hip_graph.cpp`
  `hipStreamEndCapture_common()` around line 1230 (relevant to 5.C)
- Event record capture path: `projects/clr/hipamd/src/hip_event.cpp`
  `hipEventRecord_common()` (relevant to 5.C)
- April 2026 host-launch optimization commit:
  `ROCm/rocm-systems@5d1ed85c` "clr: pre-compute graph segment
  dependency at instantiate (#4634)"

### Local cloned source (for further analysis)

`/home/chcai/rocm_source/rocm-systems/` — `develop` branch sparse-
checkout of `projects/clr/`.  CHANGELOG quoted in §3 is the
`projects/clr/CHANGELOG.md` from this tree.

### Tested ROCm tarballs (already extracted to tmpfs on reserved nodes)

- `/dev/shm/rocm_713` (on jids 4652, 5093) — HIP 7.13.26193-d342d97fde
- `/dev/shm/rocm_714` (on jid 4975) — HIP 7.13.26194-39213316d2
- NFS copies: `/apps/chcai/rocm_71{3,4}/`

### Reference GitHub issues / PRs

- `ROCm/clr#77` — "[Issue]: No performance improvement using hipGraph"
  (AMD developers acknowledge the host-overhead issue; focused on MI300)
- `ROCm/hip#3876` — "operation not permitted when stream is capturing"
  (thread-local capture mode; related to T0' switch)
- `pytorch/pytorch#177309` — same class of capture-mode bug
- `ROCm/rocm-systems#2380` — events recorded inside graphs
  (acknowledged fixed in "HIP 7.2.53150 and ROCm 7.13" per AMD)
- `ROCm/rocm-systems` PRs #2177 / #3195 — SWDEV-579356 ThreadLocal
  capture-mode fix, NOT in any shipping ROCm as of Feb 2026
- `ROCm/rocm-systems` PR #4634 (commit `5d1ed85c`) — host-launch
  optimization (§3)
- `ROCm/rocm-systems` commit `09b8ca57b3bc339d08861b7d9bf3b2645e57afc7`
  — original SWDEV-541096 commit that introduced the buggy
  `AddExternalEventWaitNode`

---

*End of cudagraph_analysis.md*
