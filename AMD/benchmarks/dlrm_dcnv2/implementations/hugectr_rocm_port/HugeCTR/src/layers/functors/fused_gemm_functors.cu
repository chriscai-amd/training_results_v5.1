#include "hip/hip_runtime.h"
#include <cstring>  // memset (ROCm port)
#include <array>
#include <mutex>
#include <hipblas/hipblas.h>  // hipblasGemmEx fallback (ROCm port)
/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <layers/functors/fused_gemm_functors.hpp>

namespace HugeCTR {

// ROCm port: post-GEMM epilogue kernels used by the GemmFunctor fallback when
// hipBLASLt has no kernel for our shape/epilogue (MultiCross v2 needs BIAS at
// fprop and BGRADA at bprop). The plain hipblasGemmEx path runs first; these
// kernels then apply the epilogue contribution.

// Type-safe FP16 <-> FP32 conversions (static_cast<float>(__half) is not a
// valid HIP intrinsic; use __half2float / __float2half).
template <typename T>
__device__ inline float to_f32(T x) { return static_cast<float>(x); }
template <>
__device__ inline float to_f32<__half>(__half x) { return __half2float(x); }

template <typename T>
__device__ inline T from_f32(float x) { return static_cast<T>(x); }
template <>
__device__ inline __half from_f32<__half>(float x) { return __float2half(x); }

template <typename T>
__global__ void add_bias_per_row_kernel(T* D, const T* __restrict__ bias, int m, int n) {
  // D is col-major m x n. Bias has length m, broadcast across the n columns.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < m && j < n) {
    size_t off = i + static_cast<size_t>(j) * m;
    float v = to_f32<T>(D[off]) + to_f32<T>(bias[i]);
    if constexpr (std::is_same<T, __half>::value) {
      constexpr float kFp16Max = 65504.0f;
      if (v > kFp16Max) v = kFp16Max;
      else if (v < -kFp16Max) v = -kFp16Max;
    }
    D[off] = from_f32<T>(v);
  }
}

template <typename T>
__global__ void reduce_sum_columns_kernel(const T* __restrict__ A, T* dbias, int m, int k) {
  // BGRADA semantics: dbias is the sum over the GEMM's contracted dimension
  // (k -- not the output cols n). For a GEMM D = op(A) * op(B), the bias
  // gradient w.r.t. the upstream fprop bias (added to D's rows) equals the
  // sum of A across the contracted axis k. Length of dbias = m (= rows of D).
  // Col-major A storage: shape (m x k), A[i, j] at offset i + j*m.
  // FP32 accumulation, then clamp before FP16 store -- otherwise per-rank
  // sums can exceed FP16 max (65504) at multi-GPU shapes and become inf,
  // which then poisons Adagrad's accumulator (sqrt(inf^2) = inf -> NaN).
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= m) return;
  float sum = 0.0f;
  for (int j = 0; j < k; ++j) {
    sum += to_f32<T>(A[i + static_cast<size_t>(j) * m]);
  }
  if constexpr (std::is_same<T, __half>::value) {
    // ROCm port: clamp at FP16 max so single-rank dbias never becomes inf.
    // We do NOT pre-divide by N: HugeCTR's loss already divides by the global
    // batch (per_rank_batch * num_gpus), so per-rank dbias is already 1/N of
    // the single-GPU equivalent and the subsequent ncclSum gives the right
    // value. Only need to guard against the edge case where a single per-rank
    // dbias element overflows due to scaler / per-sample gradient amplification.
    constexpr float kFp16Max = 65504.0f;
    if (sum > kFp16Max) sum = kFp16Max;
    else if (sum < -kFp16Max) sum = -kFp16Max;
  }
  dbias[i] = from_f32<T>(sum);
}

template <typename T>
inline void launch_add_bias_per_row(T* D, const T* bias, int m, int n, hipStream_t s) {
  if (m == 0 || n == 0 || bias == nullptr) return;
  dim3 block(16, 16, 1);
  dim3 grid((m + 15) / 16, (n + 15) / 16, 1);
  add_bias_per_row_kernel<T><<<grid, block, 0, s>>>(D, bias, m, n);
}

template <typename T>
inline void launch_reduce_sum_columns(const T* A, T* dbias, int m, int k, hipStream_t s) {
  if (m == 0 || dbias == nullptr) return;
  dim3 block(256, 1, 1);
  dim3 grid((m + 255) / 256, 1, 1);
  reduce_sum_columns_kernel<T><<<grid, block, 0, s>>>(A, dbias, m, k);
}


template <typename T>
void CublasDesc<T>::set_fprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                                   hipblasOperation_t op_a, hipblasOperation_t op_b,
                                   hipblasLtOrder_t order, bool enable_tf32_compute,
                                   const T* bias_ptr, Activation_t act, T* mask_out_ptr) {
  if (order == HIPBLASLT_ORDER_ROW) {
    row_major = true;
    std::swap(dims_a, dims_b);
    std::swap(op_a, op_b);
    std::reverse(dims_a.begin(), dims_a.end());
    std::reverse(dims_b.begin(), dims_b.end());
    // HIPBLASLT_ORDER_ROW cannot be combined with HIPBLASLT_EPILOGUE_BIAS, so this workaround is
    // needed. It treats the row-major matrix as a transpose of the col-major matrix.
  } else {
    row_major = false;
  }

  hipblasComputeType_t compute_type =
      enable_tf32_compute ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F;
  HCTR_LIB_THROW(hipblasLtMatmulDescCreate(&cublas_op_desc, compute_type, HIP_R_32F));

  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                                sizeof(op_a)));
  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                                sizeof(op_b)));

  size_t cublas_rows_c = op_a == HIPBLAS_OP_N ? dims_a[0] : dims_a[1];
  size_t cublas_cols_c = op_b == HIPBLAS_OP_N ? dims_b[1] : dims_b[0];

  epilogue = mask_out_ptr != nullptr ? HIPBLASLT_EPILOGUE_RELU_AUX : HIPBLASLT_EPILOGUE_RELU;
  if (bias_ptr != nullptr) {
    epilogue = hipblasLtEpilogue_t((int)epilogue | (int)HIPBLASLT_EPILOGUE_BIAS);
  }
  if (act == Activation_t::None) {
    epilogue = bias_ptr == nullptr ? HIPBLASLT_EPILOGUE_DEFAULT : HIPBLASLT_EPILOGUE_BIAS;
  }

  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue)));

  // Bias vector elements are the same type as alpha and beta when matrix D datatype is HIP_R_8I
  // and same as matrix D datatype otherwise.
  if (bias_ptr != nullptr) {
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                                  &bias_ptr, sizeof(bias_ptr)));
  }
  if (act != Activation_t::None) {
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc,
                                                  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                  &mask_out_ptr, sizeof(mask_out_ptr)));
    // relu_mask_ld must be divisible by 128 and be no less than the number of rows in the output
    // matrix.
    size_t relu_mask_ld = ((cublas_rows_c - 1) / 128 + 1) * 128;
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(
        cublas_op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &relu_mask_ld, sizeof(relu_mask_ld)));
  }

  uint32_t pointer_mode = HIPBLASLT_POINTER_MODE_HOST;
  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode)));

  hipDataType data_type = HIP_R_32F;
  if constexpr (std::is_same<T, __half>::value) {
    data_type = HIP_R_16F;
  }
  HCTR_LIB_THROW(
      hipblasLtMatrixLayoutCreate(&cublas_mat_a_desc, data_type, dims_a[0], dims_a[1], dims_a[0]));
  HCTR_LIB_THROW(
      hipblasLtMatrixLayoutCreate(&cublas_mat_b_desc, data_type, dims_b[0], dims_b[1], dims_b[0]));
  HCTR_LIB_THROW(hipblasLtMatrixLayoutCreate(&cublas_mat_c_desc, data_type, cublas_rows_c,
                                            cublas_cols_c, cublas_rows_c));
  // Cache for fallback path.
  saved_op_a = op_a;
  saved_op_b = op_b;
  saved_m = static_cast<int64_t>(cublas_rows_c);
  saved_n = static_cast<int64_t>(cublas_cols_c);
  saved_k = static_cast<int64_t>(op_a == HIPBLAS_OP_N ? dims_a[1] : dims_a[0]);
  saved_lda = static_cast<int64_t>(dims_a[0]);
  saved_ldb = static_cast<int64_t>(dims_b[0]);
  saved_ldc = static_cast<int64_t>(cublas_rows_c);
  // ROCm port: route BIAS through our fallback (hipblasGemmEx FP32 accum +
  // post-pass bias-add kernel) instead of hipBLASLt's BIAS epilogue. The
  // hipBLASLt 1.2 BIAS path on gfx950 appears to do FP16 accumulation in some
  // kernel variants, which causes 8-GPU FP16 MultiCross to NaN at per-rank
  // batch >= 1024 when the upstream gradient grows past FP16 range. Plain
  // gemmEx + manual FP32 add doesn't have this issue.
  saved_epilogue_is_plain = (epilogue == HIPBLASLT_EPILOGUE_DEFAULT) ||
                            (epilogue == HIPBLASLT_EPILOGUE_BIAS);
  saved_bias_ptr = const_cast<T*>(bias_ptr);
  saved_is_bias_epilogue = (bias_ptr != nullptr) && (act == Activation_t::None);
  saved_is_bgrada_epilogue = false;
}

template <typename T>
void CublasDesc<T>::set_bprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                                   hipblasOperation_t op_a, hipblasOperation_t op_b,
                                   hipblasLtOrder_t order, bool enable_tf32_compute, T* dbias_ptr,
                                   const T* mask_in_ptr) {
  if (order == HIPBLASLT_ORDER_ROW) {
    row_major = true;
    std::swap(dims_a, dims_b);
    std::swap(op_a, op_b);
    std::reverse(dims_a.begin(), dims_a.end());
    std::reverse(dims_b.begin(), dims_b.end());
    // HIPBLASLT_ORDER_ROW cannot be combined with HIPBLASLT_EPILOGUE_BIAS, so this workaround is
    // needed. It treats the row-major matrix as a transpose of the col-major matrix.
  } else {
    row_major = false;
  }

  hipblasComputeType_t compute_type =
      enable_tf32_compute ? HIPBLAS_COMPUTE_32F_FAST_TF32 : HIPBLAS_COMPUTE_32F;
  HCTR_LIB_THROW(hipblasLtMatmulDescCreate(&cublas_op_desc, compute_type, HIP_R_32F));

  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op_a,
                                                sizeof(op_a)));
  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op_b,
                                                sizeof(op_b)));
  size_t cublas_rows_c = op_a == HIPBLAS_OP_N ? dims_a[0] : dims_a[1];
  size_t cublas_cols_c = op_b == HIPBLAS_OP_N ? dims_b[1] : dims_b[0];

  if (mask_in_ptr != nullptr) {
    bool use_bgrad_bprop = dbias_ptr != nullptr;
    epilogue = use_bgrad_bprop ? HIPBLASLT_EPILOGUE_DRELU_BGRAD : HIPBLASLT_EPILOGUE_DRELU;
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)));
    if (use_bgrad_bprop) {
      HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(
          cublas_op_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias_ptr, sizeof(dbias_ptr)));
    }
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc,
                                                  HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                                  &mask_in_ptr, sizeof(mask_in_ptr)));
    size_t relu_mask_ld = ((cublas_rows_c - 1) / 128 + 1) * 128;
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(
        cublas_op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, &relu_mask_ld, sizeof(relu_mask_ld)));
  } else {
    bool use_bgrad_fuse_a = dbias_ptr != nullptr;
    epilogue = use_bgrad_fuse_a ? HIPBLASLT_EPILOGUE_BGRADA : HIPBLASLT_EPILOGUE_DEFAULT;
    HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE,
                                                  &epilogue, sizeof(epilogue)));
    if (use_bgrad_fuse_a) {
      HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(
          cublas_op_desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &dbias_ptr, sizeof(dbias_ptr)));
    }
  }

  uint32_t pointer_mode = HIPBLASLT_POINTER_MODE_HOST;
  HCTR_LIB_THROW(hipblasLtMatmulDescSetAttribute(cublas_op_desc, HIPBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode)));

  hipDataType data_type = HIP_R_32F;
  if constexpr (std::is_same<T, __half>::value) {
    data_type = HIP_R_16F;
  }

  HCTR_LIB_THROW(
      hipblasLtMatrixLayoutCreate(&cublas_mat_a_desc, data_type, dims_a[0], dims_a[1], dims_a[0]));
  HCTR_LIB_THROW(
      hipblasLtMatrixLayoutCreate(&cublas_mat_b_desc, data_type, dims_b[0], dims_b[1], dims_b[0]));

  HCTR_LIB_THROW(hipblasLtMatrixLayoutCreate(&cublas_mat_c_desc, data_type, cublas_rows_c,
                                            cublas_cols_c, cublas_rows_c));
  // Cache for fallback path.
  saved_op_a = op_a;
  saved_op_b = op_b;
  saved_m = static_cast<int64_t>(cublas_rows_c);
  saved_n = static_cast<int64_t>(cublas_cols_c);
  saved_k = static_cast<int64_t>(op_a == HIPBLAS_OP_N ? dims_a[1] : dims_a[0]);
  saved_lda = static_cast<int64_t>(dims_a[0]);
  saved_ldb = static_cast<int64_t>(dims_b[0]);
  saved_ldc = static_cast<int64_t>(cublas_rows_c);
  // ROCm port: route BGRADA through our fallback (hipblasGemmEx FP32 accum +
  // post-pass column-sum kernel) instead of hipBLASLt's BGRADA epilogue --
  // same reason as the fprop BIAS path: hipBLASLt 1.2 / gfx950 appears to
  // accumulate the bias gradient in FP16 for some kernel variants, which
  // overflows for the multi-GPU MultiCross shapes (per-rank batch >= 1024).
  saved_epilogue_is_plain = (epilogue == HIPBLASLT_EPILOGUE_DEFAULT) ||
                            (epilogue == HIPBLASLT_EPILOGUE_BGRADA);
  saved_bias_ptr = dbias_ptr;
  saved_is_bias_epilogue = false;
  saved_is_bgrada_epilogue = (epilogue == HIPBLASLT_EPILOGUE_BGRADA);
}

template <typename T>
CublasDesc<T>::~CublasDesc() {
  hipblasLtMatmulDescDestroy(cublas_op_desc);
  hipblasLtMatrixLayoutDestroy(cublas_mat_a_desc);
  hipblasLtMatrixLayoutDestroy(cublas_mat_b_desc);
  hipblasLtMatrixLayoutDestroy(cublas_mat_c_desc);
}

template <typename T>
void CublasAlgo<T>::init_algorithm(const CublasDesc<T>& cublas_desc,
                                   hipblasLtHandle_t cublaslt_handle) {
  HCTR_LIB_THROW(hipblasLtMatmulPreferenceCreate(&cublas_preference));

  HCTR_LIB_THROW(hipMalloc(&cublaslt_workspace, cublaslt_workspace_size));
  HCTR_LIB_THROW(hipblasLtMatmulPreferenceSetAttribute(
      cublas_preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &cublaslt_workspace_size,
      sizeof(cublaslt_workspace_size)));

// HugeCTR ROCm port: hipBLASLt's preference API doesn't expose POINTER_MODE_MASK
// or EPILOGUE_MASK -- those are cuBLASLt-only knobs that filter the heuristic
// search. The default preference works; the heuristic still picks an algorithm
// compatible with our matmul descriptor's epilogue.
#if 0
  uint32_t pointer_mode = HIPBLASLT_POINTER_MODE_HOST;
  HCTR_LIB_THROW(hipblasLtMatmulPreferenceSetAttribute(cublas_preference,
                                                      CUBLASLT_MATMUL_PREF_POINTER_MODE_MASK,
                                                      &pointer_mode, sizeof(pointer_mode)));
  HCTR_LIB_THROW(
      hipblasLtMatmulPreferenceSetAttribute(cublas_preference, CUBLASLT_MATMUL_PREF_EPILOGUE_MASK,
                                           &cublas_desc.epilogue, sizeof(cublas_desc.epilogue)));
#endif

  // HugeCTR ROCm port: hipBLASLt 1.2 on gfx950 may return 0 candidates for
  // exotic shapes/epilogues even though hipblasLtMatmul itself accepts a
  // null/default algo. Request 10 candidates; if any are returned, use the
  // first; otherwise fall back to a zero-initialized algo (hipblasLtMatmul
  // auto-selects when given an unconfigured algo handle).
  constexpr int kHeuristicRequest = 10;
  hipblasLtMatmulHeuristicResult_t heuristic_results[kHeuristicRequest] = {};
  int returned_res = 0;
  hipblasStatus_t st = hipblasLtMatmulAlgoGetHeuristic(
      cublaslt_handle, cublas_desc.cublas_op_desc, cublas_desc.cublas_mat_a_desc,
      cublas_desc.cublas_mat_b_desc, cublas_desc.cublas_mat_c_desc,
      cublas_desc.cublas_mat_c_desc, cublas_preference, kHeuristicRequest,
      heuristic_results, &returned_res);

  if (st == HIPBLAS_STATUS_SUCCESS && returned_res > 0) {
    algo = heuristic_results[0].algo;
    use_default_algo = false;
  } else {
    HCTR_LOG_S(WARNING, ROOT)
        << "HugeCTR ROCm port: hipBLASLt heuristic returned "
        << returned_res << " candidates (status=" << static_cast<int>(st)
        << "); will pass nullptr algo to hipblasLtMatmul." << std::endl;
    std::memset(&algo, 0, sizeof(algo));
    use_default_algo = true;
  }
  initialized = true;
}

template <typename T>
void CublasAlgo<T>::search_algorithm(const float alpha, const T* mat_a, const T* mat_b,
                                     const float beta, const T* mat_c, T* mat_d,
                                     const CublasDesc<T>& cublas_desc,
                                     hipblasLtHandle_t cublaslt_handle, hipStream_t stream) {
  if (cublas_desc.row_major) {
    std::swap(mat_a, mat_b);
  }
  if (!initialized) {
    init_algorithm(cublas_desc, cublaslt_handle);
  }
  const size_t repeat_num = 100;
  const int max_algo_count = 16;

  float shortestTime = std::numeric_limits<float>::max();
  float time;
  hipEvent_t start, stop;
  HCTR_LIB_THROW(hipEventCreate(&start));
  HCTR_LIB_THROW(hipEventCreate(&stop));

  hipblasLtMatmulHeuristicResult_t heuristic_result[max_algo_count] = {0};
  int algo_count = 0;
  HCTR_LIB_THROW(hipblasLtMatmulAlgoGetHeuristic(
      cublaslt_handle, cublas_desc.cublas_op_desc, cublas_desc.cublas_mat_a_desc,
      cublas_desc.cublas_mat_b_desc, cublas_desc.cublas_mat_c_desc, cublas_desc.cublas_mat_c_desc,
      cublas_preference, max_algo_count, heuristic_result, &algo_count));

  if (algo_count == 0) {
    HCTR_LIB_THROW(HIPBLAS_STATUS_NOT_SUPPORTED);
  }

  for (int algoIdx = 0; algoIdx < algo_count; algoIdx++) {
    hipblasStatus_t status = HIPBLAS_STATUS_SUCCESS;
    HCTR_LIB_THROW(hipEventRecord(start, stream));
    for (size_t i = 0; i < repeat_num && status == HIPBLAS_STATUS_SUCCESS; ++i) {
      status = hipblasLtMatmul(cublaslt_handle, cublas_desc.cublas_op_desc, &alpha, mat_a,
                              cublas_desc.cublas_mat_a_desc, mat_b, cublas_desc.cublas_mat_b_desc,
                              &beta, mat_c, cublas_desc.cublas_mat_c_desc, mat_d,
                              cublas_desc.cublas_mat_c_desc, &heuristic_result[algoIdx].algo,
                              cublaslt_workspace, cublaslt_workspace_size, stream);
    }
    HCTR_LIB_THROW(hipEventRecord(stop, stream));
    HCTR_LIB_THROW(hipEventSynchronize(stop));
    HCTR_LIB_THROW(hipEventElapsedTime(&time, start, stop));

    time = time / repeat_num;
    if (status != HIPBLAS_STATUS_SUCCESS) {
      continue;
    }
    if (time < shortestTime) {
      shortestTime = time;
      algo = heuristic_result[algoIdx].algo;
    }
  }

  HCTR_LIB_THROW(hipEventDestroy(start));
  HCTR_LIB_THROW(hipEventDestroy(stop));
}

template <typename T>
CublasAlgo<T>::~CublasAlgo() {
  hipFree(cublaslt_workspace);
  hipblasLtMatmulPreferenceDestroy(cublas_preference);
}

template <typename T>
void GemmFunctor<T>::operator()(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                                const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                                const CublasAlgo<T>& cublas_algo, hipblasLtHandle_t cublaslt_handle,
                                hipStream_t stream) {
  if (cublas_desc.row_major) {
    std::swap(mat_a, mat_b);
  }
  // ROCm port: hipBLASLt 1.2 on gfx950 either returns 0 heuristic candidates
  // for, or returns runtime-failing algos for, the MultiCross v2 GEMM shapes.
  // For plain GEMMs (DEFAULT epilogue: no fused ReLU/bias/aux), always route
  // through hipblasGemmEx -- functionally equivalent and known to work on
  // gfx950 across all the shapes we hit. hipBLASLt remains the path for fused
  // GEMMs (epilogue != DEFAULT, used by the unfused-MLP InnerProduct + bias
  // and by FusedFCLayer).
  if (cublas_desc.saved_epilogue_is_plain || cublas_algo.use_default_algo) {
    // The fallback emulates DEFAULT, BIAS (fprop bias add), and BGRADA
    // (bprop bias gradient = column sum of A). Other epilogues -- DRELU /
    // DRELU_BGRAD / RELU_AUX / BIAS+RELU -- still need a dedicated kernel
    // path; surface a clear error if hit. The DLRM-DCNv2 reference graph
    // (MultiCross v2 + InnerProduct MLPs) only requires the three above.
    const bool fallback_can_emulate =
        cublas_desc.saved_epilogue_is_plain ||
        cublas_desc.saved_is_bias_epilogue ||
        cublas_desc.saved_is_bgrada_epilogue;
    if (!fallback_can_emulate) {
      HCTR_OWN_THROW(
          Error_t::WrongInput,
          std::string("GemmFunctor fallback hit non-DEFAULT/BIAS/BGRADA epilogue ") +
              std::to_string(static_cast<int>(cublas_desc.epilogue)) +
              " (mask/aux/relu); plain hipblasGemmEx cannot emulate this."
              " Add a dedicated kernel path for the missing epilogue, or keep"
              " InnerProduct-based MLP layers (which avoid the fused functor).");
    }
    // Per-device handle cache. Multi-GPU runs spawn one OMP thread per device,
    // so a thread_local handle would bind to whichever device the thread first
    // touched -- racy. Cache by current device instead.
    int dev_id = -1;
    HCTR_LIB_THROW(hipGetDevice(&dev_id));
    static std::array<hipblasHandle_t, 16> handle_cache{};
    static std::mutex handle_mu;
    hipblasHandle_t blas_handle = nullptr;
    {
      std::lock_guard<std::mutex> lk(handle_mu);
      if (dev_id >= 0 && dev_id < static_cast<int>(handle_cache.size()) &&
          handle_cache[dev_id] != nullptr) {
        blas_handle = handle_cache[dev_id];
      } else {
        HCTR_LIB_THROW(hipblasCreate(&blas_handle));
        if (dev_id >= 0 && dev_id < static_cast<int>(handle_cache.size())) {
          handle_cache[dev_id] = blas_handle;
        }
      }
    }
    HCTR_LIB_THROW(hipblasSetStream(blas_handle, stream));
    // Use hipblasGemmEx so the FP16 path also gets FP32 accumulation
    // (hipblasHgemm accumulates in FP16 -> overflow/NaN at non-trivial scale).
    if constexpr (std::is_same<T, float>::value) {
      HCTR_LIB_THROW(hipblasGemmEx(
          blas_handle, cublas_desc.saved_op_a, cublas_desc.saved_op_b,
          static_cast<int>(cublas_desc.saved_m), static_cast<int>(cublas_desc.saved_n),
          static_cast<int>(cublas_desc.saved_k), &alpha,
          mat_a, HIP_R_32F, static_cast<int>(cublas_desc.saved_lda),
          mat_b, HIP_R_32F, static_cast<int>(cublas_desc.saved_ldb),
          &beta,
          mat_d, HIP_R_32F, static_cast<int>(cublas_desc.saved_ldc),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    } else {
      // FP16 inputs/output, FP32 accumulation, host-side FP32 alpha/beta.
      HCTR_LIB_THROW(hipblasGemmEx(
          blas_handle, cublas_desc.saved_op_a, cublas_desc.saved_op_b,
          static_cast<int>(cublas_desc.saved_m), static_cast<int>(cublas_desc.saved_n),
          static_cast<int>(cublas_desc.saved_k), &alpha,
          mat_a, HIP_R_16F, static_cast<int>(cublas_desc.saved_lda),
          mat_b, HIP_R_16F, static_cast<int>(cublas_desc.saved_ldb),
          &beta,
          mat_d, HIP_R_16F, static_cast<int>(cublas_desc.saved_ldc),
          HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
    if (cublas_desc.saved_is_bias_epilogue && cublas_desc.saved_bias_ptr) {
      launch_add_bias_per_row<T>(mat_d, reinterpret_cast<const T*>(cublas_desc.saved_bias_ptr),
                                 static_cast<int>(cublas_desc.saved_m),
                                 static_cast<int>(cublas_desc.saved_n), stream);
    }
    if (cublas_desc.saved_is_bgrada_epilogue && cublas_desc.saved_bias_ptr) {
      // Sum over the contracted (k) dim, not the output (n) dim.
      launch_reduce_sum_columns<T>(mat_a, reinterpret_cast<T*>(cublas_desc.saved_bias_ptr),
                                   static_cast<int>(cublas_desc.saved_m),
                                   static_cast<int>(cublas_desc.saved_k), stream);
    }
    return;
  }
  HCTR_LIB_THROW(hipblasLtMatmul(
      cublaslt_handle, cublas_desc.cublas_op_desc, &alpha, mat_a, cublas_desc.cublas_mat_a_desc,
      mat_b, cublas_desc.cublas_mat_b_desc, &beta, mat_c, cublas_desc.cublas_mat_c_desc, mat_d,
      cublas_desc.cublas_mat_c_desc, &cublas_algo.algo, cublas_algo.cublaslt_workspace,
      cublas_algo.cublaslt_workspace_size, stream));
}

// HugeCTR ROCm port: explicit instantiations at end-of-file (the upstream
// header had `template class Foo<float>;` BEFORE the function definitions,
// which amdclang silently no-ops; emitting them here ensures all symbols land).
template struct CublasDesc<float>;
template struct CublasDesc<__half>;
template struct CublasAlgo<float>;
template struct CublasAlgo<__half>;
template struct GemmFunctor<float>;
template struct GemmFunctor<__half>;

}  // namespace HugeCTR
