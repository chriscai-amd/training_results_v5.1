// Phase 14q (2026-05-14): ROCm 7.12-build stub for CK-Tile MLP kernels.
//
// CK-Tile API drift between ROCm 7.2.x and 7.11+:
//   - 7.0/7.1   CShuffleEpilogueProblem with NO `MemoryOperation` param
//   - 7.2/7.2.1 ADDED `memory_operation_enum MemoryOperation_` as required param #17
//   - 7.11/7.12 REMOVED `MemoryOperation` again (now driven from elsewhere)
//   - 7.12      ADDED `DoubleSmemBuffer_` as new trailing param
//
// Rather than maintain N codebases, this stub returns hipErrorInvalidValue
// from every fused-MLP entry point — callers fall back to hipBLASLt. Mirrors
// the Phase 14p stub we used in 7.2.x when we hit a similar drift. Transpose
// and bit-pack mask helpers (pure HIP, no CK-Tile dep) remain functional.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

extern "C" hipError_t hctr_cktile_gemm_bias_relu_fp16(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, hipStream_t) { return hipErrorInvalidValue; }

extern "C" hipError_t hctr_cktile_gemm_bias_fp16(
    const __half*, const __half*, const __half*, __half*,
    int, int, int, hipStream_t) { return hipErrorInvalidValue; }

extern "C" hipError_t hctr_cktile_gemm_plain_fp16(
    const __half*, const __half*, __half*,
    int, int, int, hipStream_t) { return hipErrorInvalidValue; }

extern "C" hipError_t hctr_cktile_gemm_dgrad_fp16(
    const __half*, const __half*, __half*,
    int, int, int, hipStream_t) { return hipErrorInvalidValue; }

extern "C" hipError_t hctr_cktile_init_dgrad_zero_bias(int) {
    return hipErrorInvalidValue;
}

__global__ void cktile_transpose_weight_kernel(
    const __half* __restrict__ d_W, __half* __restrict__ d_WT, int K, int N) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    int n = blockIdx.y * blockDim.y + threadIdx.y;
    if (k < K && n < N) d_WT[n * K + k] = d_W[k * N + n];
}

extern "C" hipError_t hctr_cktile_transpose_weight_fp16(
    const __half* d_W_row_major, __half* d_W_col_major, int K, int N, hipStream_t stream) {
    if (K < 1 || N < 1) return hipErrorInvalidValue;
    constexpr int TX = 16, TY = 16;
    dim3 block(TX, TY);
    dim3 grid((K + TX - 1) / TX, (N + TY - 1) / TY);
    hipLaunchKernelGGL(cktile_transpose_weight_kernel, grid, block, 0, stream,
                       d_W_row_major, d_W_col_major, K, N);
    return hipGetLastError();
}

__global__ void cktile_pack_relu_mask_kernel(
    const __half* __restrict__ top, uint8_t* __restrict__ aux,
    int M_hctr, int N_hctr, int aux_ld) {
    int byte_i = blockIdx.x * blockDim.x + threadIdx.x;
    int j      = blockIdx.y * blockDim.y + threadIdx.y;
    int max_byte = (N_hctr + 7) / 8;
    if (byte_i >= max_byte || j >= M_hctr) return;
    uint8_t mask = 0;
    int base_n = byte_i * 8;
    #pragma unroll
    for (int b = 0; b < 8; ++b) {
        int n = base_n + b;
        if (n >= N_hctr) break;
        size_t off = static_cast<size_t>(j) * N_hctr + n;
        if (__half2float(top[off]) > 0.0f) mask |= static_cast<uint8_t>(1u << b);
    }
    aux[static_cast<size_t>(byte_i) + static_cast<size_t>(j) * aux_ld] = mask;
}

extern "C" hipError_t hctr_cktile_pack_relu_mask_fp16(
    const __half* d_top, void* d_aux, int M, int N, int aux_ld, hipStream_t stream) {
    if (M < 1 || N < 1 || aux_ld < (N + 7) / 8) return hipErrorInvalidValue;
    constexpr int TX = 8, TY = 32;
    int max_byte = (N + 7) / 8;
    dim3 block(TX, TY);
    dim3 grid((max_byte + TX - 1) / TX, (M + TY - 1) / TY);
    hipLaunchKernelGGL(cktile_pack_relu_mask_kernel, grid, block, 0, stream,
                       d_top, reinterpret_cast<uint8_t*>(d_aux), M, N, aux_ld);
    return hipGetLastError();
}
