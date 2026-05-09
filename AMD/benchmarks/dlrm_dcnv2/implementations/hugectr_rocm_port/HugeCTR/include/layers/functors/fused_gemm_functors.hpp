#include "hip/hip_runtime.h"
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
#pragma once

#include <hipblaslt/hipblaslt.h>
#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <common.hpp>

namespace HugeCTR {

template <typename T>
struct CublasDesc {
  hipblasLtMatmulDesc_t cublas_op_desc = NULL;
  hipblasLtMatrixLayout_t cublas_mat_a_desc = NULL;
  hipblasLtMatrixLayout_t cublas_mat_b_desc = NULL;
  hipblasLtMatrixLayout_t cublas_mat_c_desc = NULL;
  hipblasLtEpilogue_t epilogue;
  bool row_major = true;

  // ROCm port: cache the column-major GEMM dims and ops so the GemmFunctor
  // fallback path (plain hipblasGemmEx, used when hipBLASLt has no kernel
  // for our shape/epilogue) doesn't need to query the matrix-layout API.
  // Set in set_fprop_attr / set_bprop_attr.
  hipblasOperation_t saved_op_a = HIPBLAS_OP_N;
  hipblasOperation_t saved_op_b = HIPBLAS_OP_N;
  int64_t saved_m = 0;
  int64_t saved_n = 0;
  int64_t saved_k = 0;
  int64_t saved_lda = 0;
  int64_t saved_ldb = 0;
  int64_t saved_ldc = 0;
  bool saved_epilogue_is_plain = true;  // true when epilogue == DEFAULT
  // For HIPBLASLT_EPILOGUE_BIAS / BGRADA fallback emulation. The bias_ptr
  // points to a per-row vector of length saved_m (col-major view of D).
  // For BIAS (fprop): post-GEMM, D[i,j] += bias[i].
  // For BGRADA (bprop): post-GEMM, bias[i] = sum_j A[i,j] (length saved_m).
  void* saved_bias_ptr = nullptr;
  bool saved_is_bias_epilogue = false;     // BIAS epilogue (post-GEMM bias add)
  bool saved_is_bgrada_epilogue = false;   // BGRADA epilogue (post-GEMM bias gradient)

  void set_fprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                      hipblasOperation_t op_a, hipblasOperation_t op_b, hipblasLtOrder_t order,
                      bool enable_tf32_compute, const T* bias_ptr = nullptr,
                      Activation_t act = Activation_t::None, T* mask_out_ptr = nullptr);
  void set_bprop_attr(std::vector<size_t> dims_a, std::vector<size_t> dims_b,
                      hipblasOperation_t op_a, hipblasOperation_t op_b, hipblasLtOrder_t order,
                      bool enable_tf32_compute, T* dbias_ptr = nullptr,
                      const T* mask_in_ptr = nullptr);

  ~CublasDesc();
};

template <typename T>
struct CublasAlgo {
  size_t cublaslt_workspace_size = 1024 * 1024 * 8;
  void* cublaslt_workspace;
  hipblasLtMatmulAlgo_t algo;
  hipblasLtMatmulPreference_t cublas_preference = NULL;
  bool initialized = false;
  // ROCm port: set true when init_algorithm couldn't get a heuristic match;
  // GemmFunctor::operator() then passes nullptr algo to hipblasLtMatmul.
  bool use_default_algo = false;

  void init_algorithm(const CublasDesc<T>& cublas_desc, hipblasLtHandle_t cublaslt_handle);

  void search_algorithm(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                        const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                        hipblasLtHandle_t cublaslt_handle, hipStream_t stream);
  ~CublasAlgo();
};

template <typename T>
struct GemmFunctor {
  // D = alpha*(A*B) + beta*(C),
  void operator()(const float alpha, const T* mat_a, const T* mat_b, const float beta,
                  const T* mat_c, T* mat_d, const CublasDesc<T>& cublas_desc,
                  const CublasAlgo<T>& cublas_algo, hipblasLtHandle_t cublaslt_handle,
                  hipStream_t stream);
};

// ROCm port: explicit-instantiation *declarations* (definitions live only in
// fused_gemm_functors.cu's end-of-file block).
extern template class CublasDesc<float>;
extern template class CublasDesc<__half>;

extern template class CublasAlgo<float>;
extern template class CublasAlgo<__half>;

extern template class GemmFunctor<float>;
extern template class GemmFunctor<__half>;
}  // namespace HugeCTR