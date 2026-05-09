# DLRM-DCNv2 on AMD ROCm

A research port of NVIDIA's HugeCTR-based MLPerf DLRM-DCNv2 submission to AMD
Instinct (MI350X / `gfx950`) under ROCm 7.2.

| Implementation | Path | Status |
|---|---|---|
| HugeCTR (hipified) | [`implementations/hugectr_rocm_port/`](implementations/hugectr_rocm_port/) | Working single-node 8 × MI350X (FP32 + real DCN-v2; FP16 mixed + InnerProduct substitute). FP16 + 8 GPU + real DCN-v2 NaN under investigation. |

This branch is **not** an official MLPerf v5.1 submission — there is no
DLRM submission from AMD or any vendor on Instinct hardware in MLPerf v5.1.
This is published as a reproducible reference for further work.

See [`implementations/hugectr_rocm_port/README.md`](implementations/hugectr_rocm_port/README.md)
for build / run instructions and the full gap analysis vs. NVIDIA's
B200 submission (`NVIDIA/benchmarks/dlrm_dcnv2/`).
