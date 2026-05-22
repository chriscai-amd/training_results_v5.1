# HCTR ROCm port — runtime environment variable reference

Inventory of every env var read by the HugeCTR ROCm port at runtime. Paths are relative to the port root (`implementations/hugectr_rocm_port/`). Source-of-truth columns cite the `getenv` site for C++ vars and the wrapper site for shell-only vars.

Conventions:
- **Default** is what the var resolves to when unset. `—` means no default (the feature is off / the code path skipped).
- "wrapper" defaults come from `scripts/run_712_match.sh` (the main launch wrapper).
- Vars listed under **§I. Notable gotchas** are common foot-guns documented separately.

---

## A. Run control (batch / iters / display / LR / scaler / precision)

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_NGPU` | 8 | `scripts/run_712_match.sh:94`, `runtime_test/python_criteo_train_ec.py:22` | GPUs per node (`--num_gpus_per_node`) |
| `HCTR_BATCH` | 55296 | `scripts/run_712_match.sh:95` | Global train batch size |
| `HCTR_EVAL_BATCH` | 1048576 | `scripts/run_712_match.sh:96` | Eval batch size |
| `HCTR_EV_SIZE` | 128 | `scripts/run_712_match.sh:97` | Embedding vector size |
| `HCTR_LR` | 0.004 | `scripts/run_712_match.sh:98` | Optimizer learning rate |
| `HCTR_MAX_ITER` | 100 | `scripts/run_712_match.sh:99` | Max training iters. **Must be ≥ 100 for valid perf** (see §I) |
| `HCTR_DISPLAY` | 10 | `scripts/run_712_match.sh:100` | `--display_interval` |
| `HCTR_PRECISION_FLAGS` | `--use_mixed_precision --scaler 16348` | `scripts/run_712_match.sh:101` | Precision CLI flags. **Production wrappers must override scaler to 1024** (see §I) |
| `HCTR_OPTIMIZER` | `adagrad` | `scripts/run_712_match.sh:152` | `--optimizer` |
| `HCTR_AUC_THRESHOLD` | 0.99 (wrapper) / 0.80275 (train.py) | `scripts/run_712_match.sh:88`, `runtime_test/nvidia_frontend/train.py:204` | AUC stop criterion |
| `HCTR_TRAIN_NUM_SAMPLES` | computed from bin filesize | `scripts/run_712_match.sh:86`, `train.py:45` | Training row count |
| `HCTR_EVAL_NUM_SAMPLES` | computed from bin filesize | `scripts/run_712_match.sh:87`, `train.py:46` | Eval row count |
| `HCTR_ENABLE_ALGO_SEARCH` | 0 | `scripts/run_712_match.sh:131` | When 0, adds `--disable_algorithm_search` (skip hipBLASLt autotune) |
| `HCTR_SLOT_SIZE` | 65536 | `train.py:63` | Slot-size floor |

---

## B. Kernel / fusion gates

These control which fused or hand-tuned kernel paths are used. Toggling them affects per-iter wall time directly.

### B.1 Bias / epilogue fusion (hipBLASLt)

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_HIPBLASLT_FUSED_EPILOGUES_FPROP` | 0 | `hugectr_hip/HugeCTR/src/layers/functors/fused_gemm_functors.cu:687` | Fuse fprop GEMM+bias+relu epilogue. **Saves ~148 µs/rank/iter** when on (Phase 17) |
| `HCTR_HIPBLASLT_FUSED_EPILOGUES_BPROP` | 0 | `fused_gemm_functors.cu:802-803` | Fuse bprop epilogues. **Must stay 0** — BGRADA Tensile pool gap on gfx950 regresses +200 µs/iter |
| `HCTR_DISABLE_BIAS` | 0 | `fused_gemm_functors.cu:1037` | Force-skip `add_bias` in fused GEMM path |
| `HCTR_DISABLE_BGRADA` | 0 | `fused_gemm_functors.cu:1101` | Force-skip BGRADA bias-grad accumulation |

### B.2 RCCL stream / scheduling

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_DEDICATED_RCCL_STREAM` | 1 | `hugectr_hip/HugeCTR/src/pybind/model_pipeline.cpp:248` | Use a dedicated stream for RCCL allreduce. Set "0" to share compute stream |
| `HCTR_RCCL_STREAM_PRIORITY` | 0 | `model_pipeline.cpp:262,357` | HIP stream priority for RCCL streams |
| `HCTR_RCCL_CU_MASK_BITS` | 0 | `hugectr_hip/HugeCTR/include/stream_event_manager.hpp:53` | CU-mask bits for RCCL streams (mp / dp / emb_ar / mlp_wgrad) |

### B.3 MLP kernel variants

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_FUSE_TOP_MLP` | 1 if BATCH ≥ 220 k else 0 | `scripts/run_712_match.sh:114`, `train.py:465` | Use fused top-MLP layer |
| `HCTR_FUSE_BOTTOM_MLP` | falls back to `HCTR_USE_FUSED_MLP` | `train.py:463` | Use fused bottom-MLP layer |
| `HCTR_USE_FUSED_MLP` | 0 | `train.py:464,466` | Master fallback fuse toggle |
| `HCTR_FUSE_WB` | True if BATCH ≥ 440 k else False | `scripts/run_712_match.sh:119`, `train.py:434` | Fuse wgrad / bias-grad |
| `HCTR_ASYNC_WGRAD` | 1 | `train.py:429` | Async wgrad in MLP layer |
| `HCTR_INNERPRODUCT_ASYNC_WGRAD` | 0 | `hugectr_hip/HugeCTR/src/layers/fully_connected_layer_half.cu:228` | Async wgrad for FP16 InnerProduct path |
| `HCTR_USE_CK_TILE_MLP` | 0 | `include/cktile_mlp_kernel.hpp:96`, `mlp_layer.cu:327` | Enable CK-tile MLP kernel |
| `HCTR_USE_CK_TILE_DGRAD` | 0 | `mlp_layer.cu:328,461` | Enable CK-tile dgrad (requires `HCTR_USE_CK_TILE_MLP=1`) |
| `HCTR_CK_TILE_FORCE` | 0 | `mlp_layer.cu:188,445` | Force CK-tile MLP code path |

### B.4 Element-wise / layer kernel selectors

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_DRELU_KERNEL` | "" (V5) | `fused_gemm_functors.cu:413` | dReLU kernel version (set "v1" for legacy) |
| `HCTR_DRELU_BGRAD_DIV` | 256.0 | `fused_gemm_functors.cu:408` | dReLU bgrad divisor |
| `HCTR_ADD_BIAS_KERNEL` | "" (V5) | `fused_gemm_functors.cu:497` | add_bias kernel version (set "v1" for legacy) |
| `HCTR_RELU_KERNEL` | "" (default) | `relu_layer.cu:137` | ReLU kernel variant (set "vec8" for opt-in vector path) |
| `HCTR_CONCAT_KERNEL` | "" (default new) | `concat_layer.cu:151` | Concat kernel variant (set "v1" for legacy) |
| `HCTR_BINARYOP_KERNEL` | "" (default) | `include/prims/mlcommon_linalg_hip.cuh:197` | Binary-op kernel variant (set "v1" for legacy) |
| `HCTR_LOOKUP_WARPS_PER_BLOCK` | 2 | `embedding/operators/generic_lookup.cuh:1363` | Warps/block for embedding lookup kernel |
| `HCTR_SKIP_DX_MEMSET` | 1 | `layers/multi_cross_layer.cu:947` | Skip `dx` memset in DCN multi-cross layer (set "0" to re-enable) |

### B.5 Pipeline overlap

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_INTRA_OVERLAP` | 1 | `train.py:339` | `train_intra_iteration_overlap` |
| `HCTR_INTER_OVERLAP` | 1 | `train.py:340` | `train_inter_iteration_overlap` |
| `HCTR_GROUPED_ALL_REDUCE` | 1 | `train.py:348` | Grouped allreduce for MLP wgrads |
| `HCTR_NUM_ITERATIONS_STATISTICS` | 20 | `train.py:349` | Stats window iters |
| `HCTR_DEFAULT_CONCURRENCY` | hardware_concurrency | `hugectr_hip/HugeCTR/src/thread_pool.cpp:29` | Thread-pool worker count |

---

## C. Memory / sharding

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_MEM_CAP` | 256 | `scripts/run_712_match.sh:128` | `--memory_cap_for_embedding` (GiB) |
| `HCTR_MEM_COMM_BW_RATIO` | 9 | `scripts/run_712_match.sh:147` | Planner `--mem_comm_bw_ratio` |
| `HCTR_MEM_COMM_WORK_RATIO` | 9 | `scripts/run_712_match.sh:148` | Planner `--mem_comm_work_ratio` |
| `HCTR_DP_SHARD_THRESH` | 0.008 | `scripts/run_712_match.sh:149` | `--dp_sharding_threshold` |
| `HCTR_SHARDING_PLAN` | `auto` | `scripts/run_712_match.sh:146` | `--sharding_plan` (`auto` or file path) |
| `HCTR_RMM_SETTABLE` | 1 | `hugectr_hip/HugeCTR/src/resource_managers/resource_manager_core.cpp:234` | Allow RMM memory-resource overrides ("0" disables) |

---

## D. Data pipeline

### D.1 Dataset selection

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_USE_MULTI_HOT` | 0 | `scripts/run_712_match.sh:68`, `train.py:59` | Multi-hot Criteo (912 B/row) vs single-hot (160 B/row) |
| `HCTR_USE_MLPERF_CRITEO` | 0 | `scripts/run_712_match.sh:69`, `train.py:76` | Use real MLPerf-published Criteo (requires `MULTI_HOT=1`) |
| `HCTR_USE_FULL_CRITEO` | 0 | `scripts/run_712_match.sh:72` | Use synthetic 24-day Criteo (requires `MULTI_HOT=1`) |
| `HCTR_USE_SUBSAMPLED_CRITEO` | 1 (in wrapper) | `scripts/run_712_match.sh:66`, `train.py:53` | Use subsampled Criteo path |
| `HCTR_USE_REAL_TABLE_SIZES` | 1 (in wrapper) | `scripts/run_712_match.sh:67`, `train.py:78` | Use real (~204 M IDs) `TABLE_SIZE_ARRAY` |
| `HCTR_DROP_TABLE_SIZE_CLAMP` | 1 if `MLPERF_CRITEO=1` else 0 | `train.py:77` | Drop the 40 M cap on the 3 largest tables |

### D.2 Reader / graph capture

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_READER_THREADS` | 4 | `train.py:382` | Data reader threads |
| `HCTR_READER_BATCHES` | 16 | `train.py:383` | Batches buffered per reader thread |
| `HCTR_USE_CUDA_GRAPH` | 1 | `train.py:337` | Capture training step as a HIP graph |
| `HCTR_RESTORE_WAIT_EXTERNAL` | 0 | `hugectr_hip/HugeCTR/src/pipeline.cpp:78,138`, `data_readers/multi_hot/async_data_reader.cpp:393,407` | External-event wait flags on restore |

### D.3 Model topology (debug / sweep)

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_USE_INNERPRODUCT_INSTEAD` | 0 | `train.py:500` | Replace MLP layer with plain InnerProduct |
| `HCTR_SKIP_MC` | 0 | `train.py:492` | Skip multi-cross (DCN) block |
| `HCTR_DCN_PROJ_DIM` | 512 | `train.py:521` | DCN projection dim |
| `HCTR_DCN_NUM_LAYERS` | 3 | `train.py:522` | DCN num layers |

### D.4 RAM staging (tracing wrappers)

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_STAGE_TRAIN_GB` | "" | `scripts/_trace_and_convert_inner.sh:50`, `scripts/_native_trace_inner.sh:29` | GiB of train data to copy to `/ramdata` |
| `HCTR_STAGE_VAL_GB` | "" | same files | GiB of val data to copy to `/ramdata` |

---

## E. Tracing / profiling

### E.1 Native Perfetto tracer

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_NATIVE_TRACE` | 0 | `hugectr_hip/HugeCTR/include/perfetto_emitter.hpp:51` | Master gate for native tracer |
| `HCTR_NATIVE_TRACE_DETAIL` | 2 | `perfetto_emitter.hpp:68` | Detail level (0=minimal, 2=full) |
| `HCTR_NATIVE_TRACE_DIR` | `/tmp` | `src/perfetto_emitter.cpp:115` | Output directory for trace JSON |
| `HCTR_NATIVE_TRACE_BEGIN` | 5 | `perfetto_emitter.cpp:339` | First iter to trace (alias `HCTR_NATIVE_TRACE_SKIP_WARMUP`) |
| `HCTR_NATIVE_TRACE_END` | begin + 5 | `perfetto_emitter.cpp:349` | Last iter to trace |
| `HCTR_NATIVE_TRACE_BASE` | 0 | `perfetto_emitter.cpp:383` | Record `iter_base` for cross-stream alignment |
| `HCTR_NATIVE_TRACE_FLUSH_EVERY` | 999999 | `perfetto_emitter.cpp:532` | Periodic flush cadence (effectively off by default) |
| `HCTR_NATIVE_TRACE_DEEP` | 0 | `src/pipeline.cpp:188` | Deep per-phase trace mode |
| `HCTR_NATIVE_TRACE_NO_PHASES` | 0 | `src/pipeline.cpp:112,211` | Disable GPU-phase emission |
| `HCTR_NATIVE_TRACE_GRAPH_NODES` | 0 | `src/graph_wrapper.cpp:123` | Emit per-graph-node events |
| `HCTR_NATIVE_TRACE_GRAPH_CLOCK` | 0 | `src/graph_wrapper.cpp:131` | Emit graph clock samples |
| `HCTR_NATIVE_TRACE_MODE` | (wrapper alias) | `scripts/_native_trace_inner.sh:80` | Wrapper feeds into `HCTR_NATIVE_TRACE` |

### E.2 ROCm tools

| Var | Default | Read at | Controls |
|---|---|---|---|
| `HCTR_ROCTX` | 0 | `hugectr_hip/HugeCTR/include/hctr_tracing.hpp:31` | Emit ROCTX range markers |
| `HCTR_PROFILE_PREFIX` | "" | `scripts/run_712_match.sh:135`, `scripts/_light_trace_inner.sh:34` | Command prefix (e.g. `rocprofv3 ...`) wrapping `python3 train.py` |
| `ROCPROF_OUTDIR` | "" | wrapper-level (`run_ttt_rocprof_ab.sh`, `_light_trace_inner.sh`) | Output dir for rocprof artifacts |

---

## F. Build / library paths

| Var | Default / behavior | Read at | Controls |
|---|---|---|---|
| `LOCAL_LIB` | unset | `scripts/run_712_match.sh:12-13` | Prepended to `LD_LIBRARY_PATH` and `PYTHONPATH`. **Without this, `run_712_match.sh` hard-overrides both vars to the installed lib path — rebuilds are silently ignored.** See §I |
| `HCTR_LIB_DIR` | `/workspace/hugectr_hip/build_rocm72/lib` | `run_b200_match.sh:17-19` | Library directory used to construct `LD_LIBRARY_PATH` / `PYTHONPATH` |
| `LD_LIBRARY_PATH` | overwritten by wrappers | `run_b200_match.sh:18`, `scripts/run_712_match.sh:12` | Library search path (overwritten — prepend with `LOCAL_LIB`) |
| `PYTHONPATH` | overwritten by wrappers | `run_b200_match.sh:19`, `scripts/run_712_match.sh:13` | Python module search path (same caveat) |
| `HIP_PLATFORM` | `amd` | `rebuild_in_container.sh:32` | Build-time only |

---

## G. ROCm / RCCL / hipBLASLt tuning

| Var | Default | Read at | Controls |
|---|---|---|---|
| `NCCL_PROTO` | `LL128` | `run_b200_match.sh:26`, `scripts/run_712_match.sh:20` | RCCL protocol (~1 % over Simple) |
| `NCCL_ALGO` | `Ring` | `run_b200_match.sh:27`, `scripts/run_712_match.sh:21` | RCCL collective algorithm |
| `NCCL_BUFFSIZE` | 8388608 (8 MiB) | `run_b200_match.sh:61`, `scripts/run_712_match.sh:55` | RCCL channel buffer size |
| `NCCL_SOCKET_IFNAME` | `lo` | `scripts/_native_trace_inner.sh:63` and other inner scripts | RCCL socket interface |
| `HIP_FORCE_DEV_KERNARG` | 1 | `run_b200_match.sh:36`, `scripts/run_712_match.sh:30` | Write kernel-arg buffers into device-visible memory |
| `DEBUG_HIP_DYNAMIC_QUEUES` | 1 | `run_b200_match.sh:51`, `scripts/run_712_match.sh:45` | Dynamic HW-queue allocation (+6–9 % throughput) |
| `DEBUG_HIP_BLOCK_SYNC` | 0 | `run_b200_match.sh:57`, `scripts/run_712_match.sh:51` | Disable host-blocking stream sync |
| `HIPBLASLT_TUNING_FILE` | (path) | `build_tune_file.sh:11` | hipBLASLt tuning database file used by `hipblaslt-bench` |
| `HIPBLASLT_LOG_FILE` | `$TRACE_DIR/hipblaslt.log` | `scripts/_trace_and_convert_inner.sh:95` | hipBLASLt log path |
| `HIPBLASLT_LOG_LEVEL` | 4 | `scripts/_trace_and_convert_inner.sh:96` | hipBLASLt log verbosity |
| `HIPBLASLT_LOG_MASK` | 0xffff | `scripts/_trace_and_convert_inner.sh:97` | hipBLASLt log topic mask |
| `OMP_NUM_THREADS` | 8 (trace wrappers) | `scripts/_*_trace_inner.sh` | OpenMP thread count |

`HIPBLASLT_TUNING_OVERRIDE_FILE` was probed and ruled out as a path to unblock the BGRADA fusion regression (Tensile pool only exposes 2 solutions); see `docs/perf_gap_350_vs_b200.md` §7.2.

---

## H. Upstream HugeCTR vars still consumed

These predate the port and live in `hugectr_hip/`. Most are debug switches that set certain pipeline stages to a no-op for bisecting.

| Var | Read at | Controls |
|---|---|---|
| `HUGECTR_LOG_LEVEL` | `core23/logger.cpp:103` | Max log level |
| `HUGECTR_LOG_TO_FILE` | `core23/logger.cpp:111` | Log to file vs stderr |
| `HUGECTR_DISABLE_OVERFLOW_CHECK` | `src/pybind/model.cpp:367` | Disable scaler overflow check |
| `HUGECTR_ROCE_GID` | `src/collectives/ib_proxy.cpp:166,213` | RoCE GID index |
| `HUGECTR_ROCE_TC` | `src/collectives/ib_proxy.cpp:209` | RoCE traffic class |
| `DENSE_UNIQUE_RATIO` | `embedding/common.cpp:859` | Embedding-distributor dense-unique ratio |
| `WGRAD_UNIQUE_RATIO` | `embedding/common.cpp:869` | Embedding wgrad unique ratio |
| `SKIP_ALL2ALL` | `embedding/hier_model_parallel_embedding.cpp:192,222` | Debug: skip all-to-all |
| `SKIP_ALLREDUCE` | `embedding/operators/communication.cpp:146` | Debug: skip allreduce |
| `SKIP_DATA_DISTRIBUTOR` | `pybind/model_pipeline.cpp:122` | Debug: skip data distributor |
| `SKIP_EMBEDDING` | `pybind/model_pipeline.cpp:151` | Debug: skip embedding lookup |
| `SKIP_BOTTOM_MLP` / `SKIP_TOP_MLP` | `pybind/model_pipeline.cpp:277,280` | Debug: skip MLP stages |
| `SKIP_H2D` | `pybind/model.cpp:1082` | Debug: skip H2D copy |
| `ONESHOT_NBLOCKS` / `ONESHOT_ALIGN_BLOCK` / `ONESHOT_MIN_BLOCK` / `ONESHOT_NCHANNELS` | `src/collectives/ib_comm_ar.cu:39-49`, `all_reduce_comm.cu:291` | IB-AR oneshot allreduce tuning |

---

## I. Notable gotchas

1. **`HCTR_PRECISION_FLAGS` scaler default is wrong for MI350X.** The wrapper default `--scaler 16348` NaNs post-numerics-fix; production wrappers must override to `--scaler 1024`.

2. **`HCTR_MAX_ITER ≤ 10` inflates throughput ~2×.** With CUDA graph on, short runs measure graph-launch latency rather than steady-state GPU wall. Use `HCTR_MAX_ITER ≥ 100` for any perf number you intend to quote; trace runs (`_native_trace_inner.sh`) use 25 iters intentionally and are not perf-valid.

3. **`LOCAL_LIB` is mandatory when testing local rebuilds.** `scripts/run_712_match.sh:12-13` hard-overwrites `LD_LIBRARY_PATH` and `PYTHONPATH` to the installed `/apps/chcai/build_rocm712/lib`. Without `LOCAL_LIB=/path/to/build/lib` prefixed, the wrapper silently loads the installed binary and any rebuild is ignored. Verify the right `libhuge_ctr_shared.so` is loaded via `/proc/<pid>/maps`.

4. **`HCTR_HIPBLASLT_FUSED_EPILOGUES_BPROP=1` regresses end-to-end.** The BGRADA Tensile pool on gfx950 / hipBLASLt 1.2 exposes only 2 solutions vs 458 for plain matmul; the bias-fused bprop falls onto the slow `MT128x256x16 MI16x16x4` family (~+200 µs/iter). Keep BPROP off; FPROP-only is the safe subset until the Tensile pool grows upstream.

5. **`HCTR_DEDICATED_RCCL_STREAM` defaults to 1 in source but to 0 in `_native_trace_inner.sh`.** Trace runs share the RCCL stream by default so timeline overlap is easier to read; production runs use a dedicated stream for the ~5 % wall win.
