#!/usr/bin/env python3
"""
HugeCTR-on-ROCm: Criteo training using the modern EmbeddingCollection API
(matches the multi-hot AsyncDataReader's data contract; old SparseEmbedding
crashes in filter_keys_per_gpu because the new reader doesn't populate the
SparseTensor fields the old API expects).
"""
import os
import sys
import traceback
import hugectr

print("=== HugeCTR-on-ROCm: Criteo via EmbeddingCollection API ===")
print(f"hugectr from: {hugectr.__file__}")

N_LABEL = 1
N_DENSE = 13
N_SPARSE = 26
SLOT_SIZE = 65536
EV_SIZE = 16
BATCHSIZE = int(os.environ.get("HCTR_BATCH", "1024"))
NGPU = int(os.environ.get("HCTR_NGPU", "1"))

# Use container paths if running inside container, else bare-metal paths.
DATA_ROOT = "/criteo" if os.path.isdir("/criteo") else "/apps/chcai/criteo_data"
TRAIN_BIN = f"{DATA_ROOT}/hugectr_bin/train_data.bin"
VAL_BIN   = f"{DATA_ROOT}/hugectr_bin/val_data.bin"
NUM_TRAIN = os.path.getsize(TRAIN_BIN) // (4 + N_DENSE * 4 + N_SPARSE * 4)
NUM_VAL   = os.path.getsize(VAL_BIN)   // (4 + N_DENSE * 4 + N_SPARSE * 4)
print(f"[ok] train rows: {NUM_TRAIN:,}   val rows: {NUM_VAL:,}")

# ---- Solver: use_embedding_collection=True is the magic flag --------------
solver = hugectr.CreateSolver(
    model_name="dlrm_criteo_rocm_ec",
    max_eval_batches=NUM_VAL // BATCHSIZE,
    batchsize_eval=BATCHSIZE,
    batchsize=BATCHSIZE,
    vvgpu=[list(range(NGPU))],
    repeat_dataset=True,
    lr=0.01,
    use_mixed_precision=False,
    i64_input_key=False,
    use_cuda_graph=False,
    use_algorithm_search=False,
    use_embedding_collection=True,
    gen_loss_summary=True,
)

optimizer = hugectr.CreateOptimizer(
    optimizer_type=hugectr.Optimizer_t.SGD,
    update_type=hugectr.Update_t.Local,
    atomic_update=True,
)

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=[TRAIN_BIN],
    eval_source=VAL_BIN,
    check_type=hugectr.Check_t.Non,
    num_samples=NUM_TRAIN,
    eval_num_samples=NUM_VAL,
    cache_eval_data=1,
    slot_size_array=[SLOT_SIZE] * N_SPARSE,
    async_param=hugectr.AsyncParam(
        num_threads=1,
        num_batches_per_thread=2,
        shuffle=False,
        multi_hot_reader=True,
        is_dense_float=True,
    ),
)

model = hugectr.Model(solver, reader, optimizer)

# ---- Input: ONE DataReaderSparseParam per slot (modern API contract) ------
model.add(hugectr.Input(
    label_dim=N_LABEL, label_name="label",
    dense_dim=N_DENSE, dense_name="dense",
    data_reader_sparse_param_array=[
        hugectr.DataReaderSparseParam(f"data{i}", 1, False, 1) for i in range(N_SPARSE)
    ],
))

# ---- EmbeddingCollection: one EmbeddingTable per slot ---------------------
embedding_tables = [
    hugectr.EmbeddingTableConfig(name=str(i), max_vocabulary_size=SLOT_SIZE, ev_size=EV_SIZE)
    for i in range(N_SPARSE)
]
ebc_config = hugectr.EmbeddingCollectionConfig(use_exclusive_keys=True)
for i in range(N_SPARSE):
    ebc_config.embedding_lookup(
        table_config=embedding_tables[i],
        bottom_name=f"data{i}",
        top_name=f"emb_vec{i}",
        combiner="sum",
    )

# Sharding plan: data-parallel across all NGPU GPUs (every table replicated).
# shard_matrix[gpu_id] = list of table_ids that GPU holds. For DP, every GPU
# holds every table.
shard_matrix = [[str(i) for i in range(N_SPARSE)] for _ in range(NGPU)]
shard_strategy = [("dp", [str(i) for i in range(N_SPARSE)])]
ebc_config.shard(shard_matrix=shard_matrix, shard_strategy=shard_strategy)
model.add(ebc_config)

# ---- Per-slot emb_vec is (B, 1, EV); reshape to 2D (B, EV) for Concat ----
for i in range(N_SPARSE):
    model.add(hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.Reshape,
        bottom_names=[f"emb_vec{i}"],
        top_names=[f"emb_flat{i}"],
        leading_dim=EV_SIZE,
    ))

model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Concat,
    bottom_names=[f"emb_flat{i}" for i in range(N_SPARSE)],
    top_names=["sparse_concat"],
))
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Concat,
    bottom_names=["dense", "sparse_concat"],
    top_names=["all_concat"],
))
for i, h in enumerate([256, 128, 1]):
    bot = "all_concat" if i == 0 else f"relu{i}"
    model.add(hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=[bot], top_names=[f"fc{i+1}"],
        num_output=h,
    ))
    if i < 2:
        model.add(hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU,
            bottom_names=[f"fc{i+1}"], top_names=[f"relu{i+1}"],
        ))
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
    bottom_names=["fc3", "label"], top_names=["loss"],
))
print("[ok] All layers added (EmbeddingCollection)")

print("\n[..] Model.compile() ...")
model.compile()
print("[ok] Model.compile() done")
print()
model.summary()

MAX_ITER  = int(os.environ.get("HCTR_MAX_ITER", "500"))
DISPLAY   = int(os.environ.get("HCTR_DISPLAY",  "50"))

print(f"\n[..] Model.fit({MAX_ITER} iters, batchsize={BATCHSIZE}) ...")
import time
t0 = time.time()
try:
    model.fit(max_iter=MAX_ITER, display=DISPLAY, eval_interval=MAX_ITER + 1,  # skip eval for clean perf
              snapshot=MAX_ITER + 1, snapshot_prefix="/tmp/hugectr_criteo_ec_")
    elapsed = time.time() - t0
    total_samples = MAX_ITER * BATCHSIZE
    print()
    print("=" * 70)
    print(f"=== HugeCTR DLRM training on Criteo on AMD MI350X ===")
    print(f"    iterations    : {MAX_ITER}")
    print(f"    batch size    : {BATCHSIZE}")
    print(f"    total samples : {total_samples:,}")
    print(f"    wall time     : {elapsed:.3f} s  (incl. setup/eval)")
    print(f"    THROUGHPUT    : {total_samples / elapsed:,.0f} samples/sec  (gross)")
    # The first iteration is warmup-heavy. Approximate steady-state via
    # HugeCTR's own per-display-window prints (parsed externally if needed).
    print("=" * 70)
except Exception as e:
    print(f"[FAIL] Model.fit(): {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)
