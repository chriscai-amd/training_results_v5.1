# Tracing & profiling

Two complementary capture paths plus an overlay that gives **NV-nsys-class
Perfetto traces with real GPU kernel names at +3–10 % iter-wall overhead**.

| Path | What it gives | Overhead | When to use |
|---|---|---:|---|
| **`rocprofv3` cold pass** (`scripts/trace_and_convert.sh`) | Every kernel symbol, hipBLASLt Tensile shapes, RCCL op-type, ROCTX phase ranges, HIP API correlation | +193 % to +370 % iter wall | One-shot per build; structural truth |
| **HCTR native trace** (`scripts/native_trace.sh`) | Per-phase GPU timing on every stream, HIP-graph aware, NV-style lane names | **+3 % (DETAIL=0) to +30 % (DETAIL=2)** | Every diagnostic run; production-near wall time |
| **Overlay** (`scripts/native_trace_overlay.py`) | Cold-pass kernel names merged onto native JSON | Same as native trace | Final Perfetto JSON for `ui.perfetto.dev` |

## Architecture (Path A + Path D)

| Mechanism | Where | Cost |
|---|---|---|
| **Path D** | `PerfettoEmitter::register_phase_for_kernel()` — `hipKernelNameRefByPtr` + demangle once, stored in `args.kernel` | Zero per launch |
| **Path A** | `native_trace_overlay.py` — cold-pass ROCTX ranges + HIP API `correlation_id` → phase→kernel map | One rocprofv3 run per build |

**Path A joiner** (two stages per ROCTX range):

1. **Correlation** (preferred): HIP API events inside the range (same TID) → matching `correlation_id` in `kernel_trace.csv`.
2. **GPU bracket** (fallback): only when stage 1 finds nothing and range ≤ 2 ms (graph-internal kernels without HIP API rows).

**Automation** (no per-callsite ROCTX edits):

- `ScopedGpuPhase` auto-pushes a matching `ScopedRange` when `HCTR_ROCTX=1`.
- Cold-pass inner script sets `HCTR_ROCTX=1` and `HCTR_NATIVE_TRACE=1` so every `ScopedGpuPhase` is active during the one-shot capture.

Path D and Path A are additive: existing `args.kernel` from Path D is preserved; overlay fills gaps.

### Achieved coverage (DLRM-DCNv2 bs=1×, HIP graph ON)

| Metric | rocprofv3-only | Native + overlay |
|---|---:|---:|
| Iter wall (5 traced iters) | ~17 ms | **~4.65 ms** |
| Overhead vs production | +311 % | **~+10 %** |
| Phase→kernel hits per rank | n/a | **140 / 140 (100 %)** |
| Intentionally skipped slices | n/a | 15 (iter envelopes, `host_api`, empty `sync_back`) |

## Workflow

```bash
# 0. Rebuild (after emitter / tracing C++ changes)
bash scripts/rebuild_in_container.sh <jobid>

# 1. One-shot cold pass (slow; rocprofv3; ROCTX + native trace ON in inner script)
HCTR_STAGE_TRAIN_GB=8 HCTR_STAGE_VAL_GB=2 \
  bash scripts/trace_and_convert.sh <jobid> 5 10 cold_v2
# → /home/chcai/rps_out/trace_cold_v2/iter5_10_*.csv

# 2. Native trace (no rocprofv3; near-production speed)
HCTR_STAGE_TRAIN_GB=8 HCTR_STAGE_VAL_GB=2 \
  HCTR_NATIVE_TRACE_DETAIL=2 \
  bash scripts/native_trace.sh <jobid> 5 10 my_native_tag
# → /home/chcai/rps_out/native_<tag>/hctr_native_trace_rank{0..7}.json

# 3. Overlay (CSV scan + JSON rewrite; seconds per rank)
bash scripts/native_overlay_all_ranks.sh \
  /home/chcai/rps_out/native_<tag> \
  /home/chcai/rps_out/trace_cold_v2/iter5_10 \
  5 8
# → overlay_rank{0..7}.json, phase_kernel_map.json
```

Open any `overlay_rank*.json` in [ui.perfetto.dev](https://ui.perfetto.dev/).

### Partial data staging

Full Criteo train bin is ~932 GB; 512 GB container tmpfs cannot hold it. Both trace scripts honor:

| Env | Effect |
|---|---|
| `HCTR_STAGE_TRAIN_GB` | Copy first N GiB of `train_data.bin` only (`dd`) |
| `HCTR_STAGE_VAL_GB` | Copy first N GiB of `val_data.bin` only |

Short profiling runs use `8` / `2` GiB. Full TTT uses host `/dev/shm` staging (see `docs/ttt.md`).

## Multi-kernel phase explosion (overlay default)

Native trace with `HCTR_USE_CUDA_GRAPH=1` emits one aggregate slice per multi-kernel phase (e.g. `[graph] network`, long `sparse_prep` buckets). The overlay **fans out** those slices into one Perfetto row per kernel:

| Stream (`tid`) | Phases exploded | Kernel names from |
|---|---|---|
| `default` | `[graph] network` (captured HIP graph) | Cold-pass `[graph] network` / `graph_network` ROCTX bracket |
| `prefetch` | Multi-kernel `sparse_prep` | Per-phase cold-pass list |
| `defaultdp` | Multi-kernel `sparse_prep` | Per-phase cold-pass list |
| `defaultmp` | Multi-kernel `[emb_a2a]` / `[emb_fwd]` | Per-phase cold-pass list |

- Children stay on the **parent stream** (graph kernels on `default`, not a separate lane).
- **Order** and **relative GPU duration** from cold pass; **wall time** scaled to native measured phase duration.
- Multi-kernel phases do **not** emit `kernel_secondary` (often contaminated by async RCCL from correlation matching).

Disable explosion:

```bash
python3 scripts/native_trace_overlay.py ... --no-explode-phases          # all off
python3 scripts/native_trace_overlay.py ... --no-explode-graph        # keep graph aggregate
python3 scripts/native_trace_overlay.py ... --no-explode-prefetch
python3 scripts/native_trace_overlay.py ... --no-explode-defaultdp
python3 scripts/native_trace_overlay.py ... --no-explode-defaultmp
```

## Tracing vs production config

Native trace scripts intentionally differ from production TTT / `run_b200_match.sh` defaults:

| Knob | Production / TTT | Native trace (`_native_trace_inner.sh`) | Why |
|---|---|---|---|
| `HCTR_PRECISION_FLAGS` | NV ref: `scaler 16348`; AMD TTT: **`scaler 1024`** | **`scaler 1024`** | AMD diverges (NaN) at 16348 post-numerics_fix; see `docs/ttt.md` |
| `HCTR_DEDICATED_RCCL_STREAM` | `1` (+3 % perf) | **`0`** | RCCL on `default` stream for simpler Perfetto lanes (no `rccl_emb_ar` / `rccl_mlp_wgrad`) |
| `HCTR_USE_CUDA_GRAPH` | `1` | `1` (default) | Production path; explosion uses cold-pass kernel list inside graph |
| `HCTR_NATIVE_TRACE_BEGIN` | n/a | default `5` | Skip warmup iters 0–4 |

Override via env when launching `native_trace.sh` (all vars passed into the container).

### First traced iter (`BEGIN`) distortion

The first in-window iter (default `iter_5`) often shows **inflated** phase durations (~24 ms iter envelope vs ~4.6 ms steady) because `GpuPhase` hipEvents can retain timestamps from untraced warmup iters 0..`BEGIN-1`. This is visible in raw native JSON and overlay output (no post-processing applied).

**Mitigation:** set `HCTR_NATIVE_TRACE_BEGIN=6` (or higher) and discard the first overlay iter when analyzing. A future emitter-side fix may recreate events at `trace_begin_iter()`; overlay does not clamp/realign iter 5.

## Why not just upgrade `rocprofv3`?

`rocprofv3` is not HIP-graph-aware on ROCm 7.2: every graph replay re-tracks all captured nodes (+193–370 % iter wall). Newer rocprofiler-sdk nightlies do not fix this for our workload. The overlay pays the rocprofv3 cost **once per build**.

## File map

| File | Role |
|---|---|
| `hugectr_hip/HugeCTR/include/perfetto_emitter.hpp` | API: `register_phase_for_kernel`, `kernel_name_from_ptr`; `ScopedGpuPhase` + auto-ROCTX |
| `hugectr_hip/HugeCTR/include/hctr_tracing.hpp` | ROCTX `ScopedRange` |
| `hugectr_hip/HugeCTR/src/perfetto_emitter.cpp` | Emitter impl, demangle cache, `ScopedGpuPhase` |
| `scripts/trace_and_convert.sh` | Outer wrapper: rocprofv3 cold pass |
| `scripts/_trace_and_convert_inner.sh` | Container body; `HCTR_ROCTX=1`, partial staging |
| `scripts/native_trace.sh` | Outer wrapper: native trace (no rocprofv3) |
| `scripts/_native_trace_inner.sh` | Container body; `scaler 1024`, `HCTR_DEDICATED_RCCL_STREAM=0` |
| `scripts/native_trace_overlay.py` | Path A joiner + multi-kernel explosion |
| `scripts/native_overlay_all_ranks.sh` | 8-rank overlay orchestration |
| `scripts/rocprofv3_to_perfetto_annotated.py` | Legacy: Perfetto JSON from rocprofv3 only |
| `scripts/rebuild_in_container.sh` | Rebuild `libhuge_ctr_shared.so` (unconditional `libaio-dev`) |

## Environment variables

### Native tracer

| Env | Default | Effect |
|---|---|---|
| `HCTR_NATIVE_TRACE` | `0` | Master gate |
| `HCTR_NATIVE_TRACE_DETAIL` | `2` | 0 ≈ +3 %; 1 ≈ +15 %; 2 ≈ +30 % (per-GEMM phases) |
| `HCTR_NATIVE_TRACE_DIR` | `/tmp` | Output dir for `hctr_native_trace_rank{R}.json` |
| `HCTR_NATIVE_TRACE_BASE` | `1` | Per-iter base hipEvent on compute stream for cross-stream offsets |
| `HCTR_NATIVE_TRACE_BEGIN` / `END` | unset | Capture iters `[BEGIN, END)` only |
| `HCTR_NATIVE_TRACE_GRAPH_NODES` | `0` | Insert hipEvent nodes into captured graph (may fail ROCm 7.2) |
| `HCTR_NATIVE_TRACE_GRAPH_CLOCK` | `0` | Clock-writer kernels in graph (~1 µs/node) |
| `HCTR_ROCTX` | `0` | ROCTX ranges; required for cold pass / auto-ROCTX in `ScopedGpuPhase` |

### Trace run wrappers

| Env | Default | Effect |
|---|---|---|
| `HCTR_STAGE_TRAIN_GB` / `HCTR_STAGE_VAL_GB` | full copy | Partial staging for short runs |
| `HCTR_DEDICATED_RCCL_STREAM` | `0` in `_native_trace_inner.sh` | `1` in production |
| `HCTR_USE_CUDA_GRAPH` | `1` | HIP graph capture for network |

## Troubleshooting

| Symptom | Fix |
|---|---|
| Empty overlay (`hits=0`) | Cold pass missing `HCTR_ROCTX=1`; check `marker_api_trace.csv` row count |
| Small `misses` | Rebuild + re-cold-pass; or add `ScopedGpuPhase` / `ScopedRange` at callsite |
| `[sync] sync_back` miss | Expected (no GPU kernel) |
| Wrong kernel on a phase (e.g. NCCL in `sparse_prep`) | Correlation noise; use explosion; check cold-pass TID filter |
| Mangled `_ZN...` in `args.kernel` | Use Path A overlay for that phase |
| `hipKernelNameRefByPtr` null | ROCm too old; check HIP runtime |
| Link error `libaio` | Run `rebuild_in_container.sh` (installs `libaio-dev` unconditionally) |
| `Loss cannot converge` on native trace | Use `scaler 1024` not `16348` (see `docs/ttt.md`) |
| Iter 5 looks ~5× longer than iter 6+ | Warmup hipEvent staleness; use `HCTR_NATIVE_TRACE_BEGIN=6` or skip iter 5 in analysis |
| `102 kernels (top: …)` still showing | Re-run overlay (explosion on by default) or old JSON |

## See also

- [`docs/ttt.md`](ttt.md) — TTT run config (`scaler 1024`, full 1 TB data)
- [`docs/numerics_fix.md`](numerics_fix.md) — convergence fix underlying stable training
- [README §3](../README.md) — bs=1× perf gap vs NV B200
