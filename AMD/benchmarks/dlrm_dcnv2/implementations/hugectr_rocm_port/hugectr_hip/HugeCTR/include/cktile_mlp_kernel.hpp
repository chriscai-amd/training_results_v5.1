// CK-Tile fused MLP kernel: PUBLIC API (no CK-Tile templates exposed).
//
// The actual implementation lives in HugeCTR/src/layers/cktile_mlp_kernel.cu
// which is compiled SEPARATELY with isolated flags (-no-fopenmp, no warp-mask
// shim, no HCTR-specific defines) so CK-Tile's WarpGemmDispatcher template
// instantiates correctly.
//
// Status (2026-05-13):
//   Phases 1-4 complete (POCs verified, see /home/chcai/cktile_*.cpp).
//   Phase 5: separate-compilation bridge in this file + cktile_mlp_kernel.cu.
//   Phase 6: e2e perf measurement after build.
//
// See README §Phase 14i.5 for full timeline.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdlib>

// Public API: extern "C" declared at GLOBAL scope so callers don't need
// namespace qualification. Implementation in cktile_mlp_kernel.cu.
extern "C" {

// Fused GEMM + bias + ReLU.
// A is M×K row-major, B is K×N column-major (matches HCTR weight layout),
// bias is 1×N broadcast, C is M×N row-major.
// Returns hipSuccess on success, hipErrorInvalidValue if shape unsupported.
hipError_t hctr_cktile_gemm_bias_relu_fp16(
    const __half* d_A, const __half* d_B, const __half* d_bias, __half* d_C,
    int M, int N, int K, hipStream_t stream);

// Fused GEMM + bias (no activation, for last MLP layer).
hipError_t hctr_cktile_gemm_bias_fp16(
    const __half* d_A, const __half* d_B, const __half* d_bias, __half* d_C,
    int M, int N, int K, hipStream_t stream);

// Plain GEMM (no bias, no activation, for dgrad/wgrad in backward pass).
hipError_t hctr_cktile_gemm_plain_fp16(
    const __half* d_A, const __half* d_B, __half* d_C,
    int M, int N, int K, hipStream_t stream);

// Transpose HCTR's K×N row-major weight into K×N col-major (stride K).
// Required because CK-Tile GEMMs above expect B in col-major (Phase 14n.2
// weight pre-transpose path). Caller allocates d_W_col_major (size K*N halves).
hipError_t hctr_cktile_transpose_weight_fp16(
    const __half* d_W_row_major, __half* d_W_col_major,
    int K, int N, hipStream_t stream);

// Phase 14n.7 (2026-05-14): bit-pack a ReLU activation mask in the format
// expected by HCTR's bprop_drelu kernels (uint8_t bytes, 1 bit per element,
// bit (row & 7) of byte at (row/8 + col * aux_ld) in cublas col-major view).
// d_top is HCTR's M×N row-major output of ReLU. d_aux is the M×N mask buffer
// (allocated by HCTR with size aux_ld * M_hctr bytes where aux_ld = ceil(N/8)).
hipError_t hctr_cktile_pack_relu_mask_fp16(
    const __half* d_top, void* d_aux, int M, int N, int aux_ld, hipStream_t stream);

// Phase 14p.2 (2026-05-14): MLP backward dgrad via CK-Tile.
// Math: d_bottom_bprop[m, k] = sum_n d_grad_top[m, n] * d_kernel[k, n]
// Layout fit: A=grad_top (M×K_dgrad row-major where K_dgrad=out_size),
//             B=kernel (kernel as N×K col-major; K=out, N=in — same memory
//             as kernel's HCTR K_orig×N_orig=in×out row-major),
//             C=bottom_bprop (M×N_dgrad row-major, N_dgrad=in_size).
// Caller passes M=batch, N_dgrad=in_size, K_dgrad=out_size.
// No explicit transpose required; uses CK-Tile zero-bias workaround
// (upstream Ds-tuple static_assert blocks bias-less GEMM, but
// bias=0 produces identical numerical result to plain GEMM).
//
// REQUIRES caller to invoke hctr_cktile_init_dgrad_zero_bias(max_n)
// BEFORE any HIP graph capture starts, where max_n is the largest
// in_size across all MLP layers — typically 1024 for DLRM-DCNv2.
// Without init, this entry point returns hipErrorInvalidValue (caller
// should fall back to hipBLAS dgrad).
hipError_t hctr_cktile_gemm_dgrad_fp16(
    const __half* d_grad_top, const __half* d_kernel, __half* d_bottom_bprop,
    int M, int N_dgrad, int K_dgrad, hipStream_t stream);

// Pre-allocate the zero-bias scratch buffer used by dgrad. Call ONCE per
// device, BEFORE HIP graph capture starts (during MLPLayer::initialize()).
// max_n = largest in_size across MLP layers (DLRM-DCNv2 = 1024).
hipError_t hctr_cktile_init_dgrad_zero_bias(int max_n);

}  // extern "C"

namespace HugeCTR {
namespace cktile {

// Inline helpers (don't pull in CK-Tile headers).
inline bool is_shape_supported(int M, int N, int K) {
    // CK-Tile config 128x128x32 MFMA32 requires N>=64 (1 N_warp_tile=32 × 2 N_warp=64),
    // K>=32 (= K_tile), and M>=64 (1 M_warp_tile=32 × 2 M_warp=64; padding handles rest).
    return (N >= 64) && (K >= 32) && (M >= 64);
}

inline bool is_enabled() {
    static const bool enabled = []() {
        const char* env = std::getenv("HCTR_USE_CK_TILE_MLP");
        return env != nullptr && std::atoi(env) == 1;
    }();
    return enabled;
}

}  // namespace cktile
}  // namespace HugeCTR
