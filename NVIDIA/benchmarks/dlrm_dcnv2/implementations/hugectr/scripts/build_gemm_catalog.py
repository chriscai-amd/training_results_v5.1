#!/usr/bin/env python3
"""
Build a GEMM shape catalog from gemm_init_log.jsonl (produced by
cublaslt_gemm_logger_shim.so during HCTR init).

Reads:    one JSONL line per cublasLtMatmul call.
Writes:   catalog JSON keyed by (M, N, K, op_a, op_b) with the
          observed dtype/compute/epilogue/ld attributes attached.
          Also writes secondary indices keyed by (M, N) for tile-shape
          based lookup from the annotation script.

Usage:
    python3 build_gemm_catalog.py <input.jsonl> [output.json]
"""
import json, sys, os
from collections import defaultdict

if len(sys.argv) < 2:
    print(__doc__); sys.exit(1)
JSONL = sys.argv[1]
OUT   = sys.argv[2] if len(sys.argv) > 2 \
        else os.path.join(os.path.dirname(JSONL), "gemm_catalog.json")

shapes = {}  # (M,N,K,op_a,op_b) -> aggregated record
by_mn  = defaultdict(list)  # (M,N) -> list of records (different K/op/epi)

with open(JSONL) as f:
    for line in f:
        d = json.loads(line)
        key = (d["M"], d["N"], d["K"], d["op_a"], d["op_b"])
        if key not in shapes:
            shapes[key] = {
                "M": d["M"], "N": d["N"], "K": d["K"],
                "op_a": d["op_a"], "op_b": d["op_b"],
                "dt_a": d["dt_a"], "dt_b": d["dt_b"],
                "dt_c": d["dt_c"], "dt_d": d["dt_d"],
                "compute": d["compute"],
                "epilogues": set(),
                "ld_a": d["ld_a"], "ld_b": d["ld_b"],
                "ld_c": d["ld_c"], "ld_d": d["ld_d"],
                "count": 0,
                "first_seq": d.get("seq", -1),
                "first_ts_ns": d.get("ts_ns", -1),
            }
        shapes[key]["epilogues"].add(d["epilogue"])
        shapes[key]["count"] += 1

# Convert sets to sorted lists for JSON
records = []
for key, rec in sorted(shapes.items()):
    rec["epilogues"] = sorted(rec["epilogues"])
    records.append(rec)

# Also: dlrm-dcnv2-specific layer hint based on (M, N, K, op)
# Bottom MLP fwd: 13->512->256->128 with batch in N
LAYER_HINTS = {
    # Bottom MLP fwd (RELU+BIAS) -- batch in N=6912/GPU
    (512, 6912, 13,   "NN"): "bot_mlp_L1_fwd",      # 13 -> 512
    (256, 6912, 512,  "NN"): "bot_mlp_L2_fwd",      # 512 -> 256
    (128, 6912, 256,  "NN"): "bot_mlp_L3_fwd",      # 256 -> 128
    # Bottom MLP bwd dgrad (op NT)
    (512, 13,   6912, "NT"): "bot_mlp_L1_bwd_dgrad",
    (256, 512,  6912, "NT"): "bot_mlp_L2_bwd_dgrad",
    (128, 256,  6912, "NT"): "bot_mlp_L3_bwd_dgrad",
    # Bottom MLP bwd wgrad (op TN)
    (256, 6912, 128,  "TN"): "bot_mlp_L3_bwd_wgrad",
    (512, 6912, 256,  "TN"): "bot_mlp_L2_bwd_wgrad",
    # (L1 wgrad: 13x6912x512 -- doesn't appear in catalog, fused into AB)
    # Top MLP fwd (RELU+BIAS), batch in N=6912
    (1024, 6912, 3456, "NN"): "top_mlp_L1_fwd",     # 3456 -> 1024
    (1024, 6912, 1024, "NN"): "top_mlp_L2_fwd",     # 1024 -> 1024
    (512,  6912, 1024, "NN"): "top_mlp_L3_fwd",     # 1024 -> 512
    (256,  6912, 512,  "NN"): "top_mlp_L4_fwd",     # 512 -> 256
    (1,    6912, 256,  "NN"): "top_mlp_L5_fwd",     # 256 -> 1 (BIAS-only)
    # Top MLP bwd dgrad (op NT)
    (1024, 1024, 6912, "NT"): "top_mlp_L2_bwd_dgrad",
    (1024, 3456, 6912, "NT"): "top_mlp_L1_bwd_dgrad",
    (512,  1024, 6912, "NT"): "top_mlp_L3_bwd_dgrad",
    (256,  512,  6912, "NT"): "top_mlp_L4_bwd_dgrad",
    (1,    256,  6912, "NT"): "top_mlp_L5_bwd_dgrad",
    # Top MLP bwd wgrad (op TN)
    (1024, 6912, 1024, "TN"): "top_mlp_L2_bwd_wgrad",
    (256,  6912, 1,    "TN"): "top_mlp_L5_bwd_wgrad",
    (1024, 6912, 512,  "TN"): "top_mlp_L3_bwd_wgrad",  # not in catalog list above; placeholder
    (3456, 6912, 1024, "TN"): "top_mlp_L1_bwd_wgrad",  # placeholder
    # Cross network (3 layers, projection_dim=512) MultiCross
    (512,  6912, 3456, "NN"): "cross_proj_fwd",        # 3456 -> 512 (cross proj down)
    (3456, 6912, 512,  "NN"): "cross_expand_fwd",      # 512 -> 3456 (cross expand up)
    (512,  3456, 6912, "NT"): "cross_proj_bwd_dgrad",
    (3456, 512,  6912, "NT"): "cross_expand_bwd_dgrad",
    (3456, 6912, 512,  "TN"): "cross_expand_bwd_wgrad",
    (512,  6912, 3456, "TN"): "cross_proj_bwd_wgrad",
}
for r in records:
    hint = LAYER_HINTS.get((r["M"], r["N"], r["K"], r["op_a"]+r["op_b"]))
    r["dlrm_layer"] = hint if hint else ""
    # GFLOPs = 2*M*N*K / 1e9
    r["gflops"]  = round(2.0 * r["M"] * r["N"] * r["K"] / 1e9, 3)
    # Estimate bytes accessed (M*K + K*N + M*N) * dtype_bytes
    dt_bytes = {"R_16F": 2, "R_16BF": 2, "R_32F": 4, "R_64F": 8,
                "R_8I": 1, "R_8U": 1, "R_32I": 4, "R_32U": 4,
                "R_8FE4M3": 1, "R_8FE5M2": 1}.get(r["dt_a"], 2)
    r["bytes_accessed"] = (r["M"] * r["K"] + r["K"] * r["N"] + r["M"] * r["N"]) * dt_bytes
    by_mn[(r["M"], r["N"])].append(r)

# Build (M,N) -> [records] index for fast lookup -- carries all enriched fields
index_by_mn = {}
for (m, n), recs in by_mn.items():
    index_by_mn[f"{m}x{n}"] = sorted(
        recs, key=lambda x: (x["op_a"] + x["op_b"], x["K"])
    )

out = {
    "schema_version": 1,
    "source_jsonl": os.path.abspath(JSONL),
    "n_records": len(records),
    "n_calls_logged": sum(r["count"] for r in records),
    "records": records,
    "by_mn": index_by_mn,
    "layer_hint_count": sum(1 for r in records if r.get("dlrm_layer")),
}

with open(OUT, "w") as f:
    json.dump(out, f, indent=2)
print(f"Wrote {OUT}")
print(f"  unique shapes: {len(records)}  total calls logged: {out['n_calls_logged']}")
print(f"  shapes with dlrm_layer hint: {out['layer_hint_count']} / {len(records)}")
print(f"  shapes per (M,N) bucket: min={min(len(v) for v in index_by_mn.values())}, "
      f"max={max(len(v) for v in index_by_mn.values())}, "
      f"avg={len(records)/len(index_by_mn):.1f}")
