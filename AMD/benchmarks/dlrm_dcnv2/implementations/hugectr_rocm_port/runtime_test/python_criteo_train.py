#!/usr/bin/env python3
"""
HugeCTR-on-ROCm: real Criteo training run on AMD MI350X.

Uses the day_0 npy data we preprocessed earlier, converted to HugeCTR's
RawAsync multi-hot binary layout. Builds a tiny DLRM-shaped network
(embedding + concat + MLP) and runs Model.fit().
"""
import os
import sys
import traceback
import hugectr

print("=== HugeCTR-on-ROCm: real Criteo training on AMD MI350X ===")
print(f"hugectr from: {hugectr.__file__}")
print()

# Match what criteo_npy_to_hugectr_bin.py wrote.
N_LABEL = 1
N_DENSE = 13
N_SPARSE = 26
SLOT_SIZE = 65536  # per-slot embedding size used for the 100k smoke
EV_SIZE = 16
MULTI_HOT = [1] * N_SPARSE  # single-hot per slot
BATCHSIZE = 1024  # hipBLASLt 1.0.7 on gfx950 lacks MFMA kernels for very small GEMMs
NGPU = 1  # start with 1 GPU; bump to 8 once it works

TRAIN_BIN = "/apps/chcai/criteo_data/hugectr_bin/train_data.bin"
VAL_BIN   = "/apps/chcai/criteo_data/hugectr_bin/val_data.bin"

assert os.path.exists(TRAIN_BIN), f"missing {TRAIN_BIN}"
assert os.path.exists(VAL_BIN),   f"missing {VAL_BIN}"

NUM_TRAIN = os.path.getsize(TRAIN_BIN) // (4 + N_DENSE * 4 + N_SPARSE * 4)
NUM_VAL   = os.path.getsize(VAL_BIN)   // (4 + N_DENSE * 4 + N_SPARSE * 4)
print(f"[ok] train rows: {NUM_TRAIN:,}   val rows: {NUM_VAL:,}")

solver = hugectr.CreateSolver(
    model_name="dlrm_criteo_rocm",
    max_eval_batches=NUM_VAL // BATCHSIZE,
    batchsize_eval=BATCHSIZE,
    batchsize=BATCHSIZE,
    vvgpu=[list(range(NGPU))],
    repeat_dataset=True,
    lr=0.01,
    use_mixed_precision=True,         # FP16 — better MFMA kernel coverage on gfx950
    scaler=1024,
    i64_input_key=False,
    use_cuda_graph=False,
    use_algorithm_search=False,
    gen_loss_summary=True,
)
print("[ok] Solver ready")

optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
print("[ok] Optimizer ready")

async_param = hugectr.AsyncParam(
    num_threads=1,
    num_batches_per_thread=2,
    shuffle=False,
    aligned_type=hugectr.Alignment_t.Auto,
    multi_hot_reader=True,
    is_dense_float=True,
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
    async_param=async_param,
)
print("[ok] DataReaderParams ready (Criteo day_0, 100k rows, single-hot)")

model = hugectr.Model(solver, reader, optimizer)
print("[ok] Model ready")

model.add(hugectr.Input(
    label_dim=N_LABEL, label_name="label",
    dense_dim=N_DENSE, dense_name="dense",
    data_reader_sparse_param_array=[
        hugectr.DataReaderSparseParam("sparse", MULTI_HOT, True, N_SPARSE),
    ],
))

model.add(hugectr.SparseEmbedding(
    embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
    workspace_size_per_gpu_in_mb=2048,  # ~26 slots * 65536 * 16 * 4B = 27 MB; oversize
    embedding_vec_size=EV_SIZE,
    combiner="sum",
    sparse_embedding_name="sparse_embedding",
    bottom_name="sparse",
    optimizer=optimizer,
))

model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Reshape,
    bottom_names=["sparse_embedding"],
    top_names=["sparse_reshape"],
    leading_dim=N_SPARSE * EV_SIZE,
))

model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Concat,
    bottom_names=["dense", "sparse_reshape"],
    top_names=["concat1"],
))

# 3-layer MLP (DLRM-shaped): hidden 256 -> 128 -> 1
for i, h in enumerate([256, 128, 1]):
    bot = "concat1" if i == 0 else f"relu{i}"
    fc  = f"fc{i+1}"
    relu = f"relu{i+1}"
    model.add(hugectr.DenseLayer(
        layer_type=hugectr.Layer_t.InnerProduct,
        bottom_names=[bot], top_names=[fc],
        num_output=h,
    ))
    if i < 2:
        model.add(hugectr.DenseLayer(
            layer_type=hugectr.Layer_t.ReLU,
            bottom_names=[fc], top_names=[relu],
        ))

model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
    bottom_names=["fc3", "label"],
    top_names=["loss"],
))
print("[ok] All layers added")

print()
print("[..] Model.compile() ...")
model.compile()
print("[ok] Model.compile() done")

print()
model.summary()

print()
print("[..] Model.fit(50 iters) on real Criteo day_0 ...")
try:
    model.fit(
        max_iter=50,
        display=10,
        eval_interval=20,
        snapshot=10000,
        snapshot_prefix="/tmp/hugectr_criteo_",
    )
    print()
    print("=== HugeCTR DLRM TRAINING ON REAL CRITEO DATA WORKED ON AMD MI350X ===")
except Exception as e:
    print(f"[FAIL] Model.fit() raised: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)
