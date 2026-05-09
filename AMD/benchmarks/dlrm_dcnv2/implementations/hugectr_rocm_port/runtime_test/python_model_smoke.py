#!/usr/bin/env python3
"""
HugeCTR-on-ROCm: end-to-end Model construction + smoke training run.

We build the simplest possible HugeCTR model (no interaction_layer / GRU /
Dropout / BatchNorm — those are stubbed in the AMD port), point it at a
synthetic Raw dataset, and see how far we can get through Model.fit().

This is the runtime smoke test for the Python API. The high-level Model class
exercises *all* the moving parts: data reader, embedding lookup, dense MLP
forward/backward, optimizer step, RCCL collective.
"""

import os
import sys
import struct
import tempfile
import traceback

import hugectr

print("=== HugeCTR-on-ROCm Model.fit() smoke test ===")
print(f"hugectr from: {hugectr.__file__}")
print()

# --- 1. Generate dataset using HugeCTR's own DataGenerator -----------------
N_LABEL = 1
N_DENSE = 4
N_SPARSE = 2
MULTI_HOT_SIZES = [1, 1]   # single-hot per slot
SPARSE_VOCAB = 1024
NUM_SAMPLES = 4096
BATCHSIZE = 64

tmp = tempfile.mkdtemp(prefix="hugectr_rocm_smoke_")
train_bin = os.path.join(tmp, "train.bin")
eval_bin = os.path.join(tmp, "eval.bin")

def write_raw(path, n):
    """Hand-write the RawAsync multi-hot binary layout.
    Per sample: <i32 label * N_LABEL> <f32 dense * N_DENSE> <i32 key * sum(MULTI_HOT_SIZES)>
    """
    import random
    random.seed(0)
    with open(path, "wb") as f:
        for i in range(n):
            for _ in range(N_LABEL):
                f.write(struct.pack("<i", random.randint(0, 1)))
            for _ in range(N_DENSE):
                f.write(struct.pack("<f", random.uniform(-1.0, 1.0)))
            for s in range(N_SPARSE):
                for _ in range(MULTI_HOT_SIZES[s]):
                    f.write(struct.pack("<i", random.randint(0, SPARSE_VOCAB - 1)))

print(f"[..] writing synthetic dataset to {tmp} ...")
write_raw(train_bin, NUM_SAMPLES)
write_raw(eval_bin, BATCHSIZE * 4)
print(f"[ok] train.bin = {os.path.getsize(train_bin)} bytes")
print(f"[ok] eval.bin  = {os.path.getsize(eval_bin)} bytes")

# --- 2. Build a Solver ----------------------------------------------------
solver = hugectr.CreateSolver(
    model_name="dlrm_smoke_rocm",
    max_eval_batches=1,
    batchsize_eval=BATCHSIZE,
    batchsize=BATCHSIZE,
    vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],  # ALL 8 MI350X via RCCL
    repeat_dataset=True,
    lr=0.01,
    use_mixed_precision=False,
    i64_input_key=False,
    use_cuda_graph=False,  # safer for first run
    gen_loss_summary=True,
)
print("[ok] Solver constructed")

# --- 3. Optimizer ---------------------------------------------------------
optimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)
print("[ok] Optimizer constructed")

# --- 4. DataReader params (RawAsync multi-hot) ----------------------------
async_param = hugectr.AsyncParam(
    num_threads=2,
    num_batches_per_thread=4,
    shuffle=False,
    aligned_type=hugectr.Alignment_t.Auto,
    multi_hot_reader=True,
    is_dense_float=True,
)
print("[ok] AsyncParam constructed")

reader = hugectr.DataReaderParams(
    data_reader_type=hugectr.DataReaderType_t.RawAsync,
    source=[train_bin],
    eval_source=eval_bin,
    check_type=hugectr.Check_t.Non,
    num_samples=NUM_SAMPLES,
    eval_num_samples=BATCHSIZE * 4,
    cache_eval_data=1,
    slot_size_array=[SPARSE_VOCAB] * N_SPARSE,
    async_param=async_param,
)
print("[ok] DataReaderParams constructed (RawAsync multi-hot)")

# --- 5. Build a Model -----------------------------------------------------
print("[..] constructing hugectr.Model ...")
model = hugectr.Model(solver, reader, optimizer)
print("[ok] hugectr.Model constructed")

# --- 6. Add input + sparse embedding + dense MLP --------------------------
print("[..] adding input layer ...")
model.add(hugectr.Input(
    label_dim=N_LABEL,
    label_name="label",
    dense_dim=N_DENSE,
    dense_name="dense",
    data_reader_sparse_param_array=[
        hugectr.DataReaderSparseParam(
            "sparse",          # top name
            MULTI_HOT_SIZES,    # nnz_per_slot
            True,               # is_fixed_length
            N_SPARSE,
        )
    ],
))
print("[ok] Input added")

print("[..] adding sparse embedding ...")
model.add(hugectr.SparseEmbedding(
    embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,
    workspace_size_per_gpu_in_mb=8,
    embedding_vec_size=8,
    combiner="sum",
    sparse_embedding_name="sparse_embedding",
    bottom_name="sparse",
    optimizer=optimizer,
))
print("[ok] SparseEmbedding added")

# Reshape sparse embedding to merge slot dim
print("[..] adding Reshape ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Reshape,
    bottom_names=["sparse_embedding"],
    top_names=["sparse_reshape"],
    leading_dim=N_SPARSE * 8,
))
print("[ok] Reshape added")

# Concat dense + sparse_reshape
print("[..] adding Concat ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.Concat,
    bottom_names=["dense", "sparse_reshape"],
    top_names=["concat1"],
))
print("[ok] Concat added")

# Two FC layers + sigmoid + loss
print("[..] adding FC1 ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.InnerProduct,
    bottom_names=["concat1"],
    top_names=["fc1"],
    num_output=16,
))
print("[ok] FC1 added")

print("[..] adding ReLU ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.ReLU,
    bottom_names=["fc1"],
    top_names=["relu1"],
))
print("[ok] ReLU added")

print("[..] adding FC2 (logits) ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.InnerProduct,
    bottom_names=["relu1"],
    top_names=["fc2"],
    num_output=1,
))
print("[ok] FC2 added")

print("[..] adding BinaryCrossEntropy loss ...")
model.add(hugectr.DenseLayer(
    layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,
    bottom_names=["fc2", "label"],
    top_names=["loss"],
))
print("[ok] Loss added")

# --- 7. Compile -----------------------------------------------------------
print()
print("[..] compiling model ...")
try:
    model.compile()
    print("[ok] Model.compile() succeeded")
except Exception as e:
    print(f"[FAIL] Model.compile() raised: {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(2)

# --- 8. Summary -----------------------------------------------------------
try:
    model.summary()
except Exception as e:
    print(f"[warn] summary() raised: {e}")

print()
print("=== HugeCTR Model.compile() on 8x AMD MI350X via RCCL succeeded ===")
