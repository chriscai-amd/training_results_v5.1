/*
 * Copyright (c) 2019-2020, NVIDIA CORPORATION.
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

#include <hipsparse.h>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace Sparse {

#define _CUSPARSE_ERR_TO_STR(err) \
  case err:                       \
    return #err;
inline const char* cusparseErr2Str(hipsparseStatus_t err) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 10100
  return hipsparseGetErrorString(status);
#else   // CUDART_VERSION
  switch (err) {
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_SUCCESS);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_NOT_INITIALIZED);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_ALLOC_FAILED);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_INVALID_VALUE);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_ARCH_MISMATCH);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_EXECUTION_FAILED);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_INTERNAL_ERROR);
    _CUSPARSE_ERR_TO_STR(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED);
    default:
      return "CUSPARSE_STATUS_UNKNOWN";
  };
#endif  // CUDART_VERSION
}
#undef _CUSPARSE_ERR_TO_STR

/** check for cusparse runtime API errors and assert accordingly */
#define CUSPARSE_CHECK(call)                                         \
  do {                                                               \
    hipsparseStatus_t err = call;                                     \
    ASSERT(err == HIPSPARSE_STATUS_SUCCESS,                           \
           "CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
           MLCommon::Sparse::cusparseErr2Str(err));                  \
  } while (0)

/** check for cusparse runtime API errors but do not assert */
#define CUSPARSE_CHECK_NO_THROW(call)                                          \
  do {                                                                         \
    hipsparseStatus_t err = call;                                               \
    if (err != HIPSPARSE_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUSPARSE call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::Sparse::cusparseErr2Str(err));                  \
    }                                                                          \
  } while (0)

/**
 * @defgroup gthr cusparse gather methods
 * @{
 */
template <typename T>
hipsparseStatus_t cusparsegthr(hipsparseHandle_t handle, int nnz, const T* vals,
                              T* vals_sorted, int* d_P, hipStream_t stream);
template <>
inline hipsparseStatus_t cusparsegthr(hipsparseHandle_t handle, int nnz,
                                     const double* vals, double* vals_sorted,
                                     int* d_P, hipStream_t stream) {
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseDgthr(handle, nnz, vals, vals_sorted, d_P,
                       HIPSPARSE_INDEX_BASE_ZERO);
}
template <>
inline hipsparseStatus_t cusparsegthr(hipsparseHandle_t handle, int nnz,
                                     const float* vals, float* vals_sorted,
                                     int* d_P, hipStream_t stream) {
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  return hipsparseSgthr(handle, nnz, vals, vals_sorted, d_P,
                       HIPSPARSE_INDEX_BASE_ZERO);
}
/** @} */

/**
 * @defgroup coo2csr cusparse COO to CSR converter methods
 * @{
 */
template <typename T>
void cusparsecoo2csr(hipsparseHandle_t handle, const T* cooRowInd, int nnz,
                     int m, T* csrRowPtr, hipStream_t stream);
template <>
inline void cusparsecoo2csr(hipsparseHandle_t handle, const int* cooRowInd,
                            int nnz, int m, int* csrRowPtr,
                            hipStream_t stream) {
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(hipsparseXcoo2csr(handle, cooRowInd, nnz, m, csrRowPtr,
                                  HIPSPARSE_INDEX_BASE_ZERO));
}
/** @} */

/**
 * @defgroup coosort cusparse coo sort methods
 * @{
 */
template <typename T>
size_t cusparsecoosort_bufferSizeExt(hipsparseHandle_t handle, int m, int n,
                                     int nnz, const T* cooRows,
                                     const T* cooCols, hipStream_t stream);
template <>
inline size_t cusparsecoosort_bufferSizeExt(hipsparseHandle_t handle, int m,
                                            int n, int nnz, const int* cooRows,
                                            const int* cooCols,
                                            hipStream_t stream) {
  size_t val;
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    hipsparseXcoosort_bufferSizeExt(handle, m, n, nnz, cooRows, cooCols, &val));
  return val;
}

template <typename T>
void cusparsecoosortByRow(hipsparseHandle_t handle, int m, int n, int nnz,
                          T* cooRows, T* cooCols, T* P, void* pBuffer,
                          hipStream_t stream);
template <>
inline void cusparsecoosortByRow(hipsparseHandle_t handle, int m, int n, int nnz,
                                 int* cooRows, int* cooCols, int* P,
                                 void* pBuffer, hipStream_t stream) {
  CUSPARSE_CHECK(hipsparseSetStream(handle, stream));
  CUSPARSE_CHECK(
    hipsparseXcoosortByRow(handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}
/** @} */

/**
 * @defgroup Gemmi cusparse gemmi operations
 * @{
 */
inline hipsparseStatus_t cusparsegemmi(
  hipsparseHandle_t handle, int m, int n, int k, int nnz, const float* alpha,
  const float* A, int lda, const float* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const float* beta, float* C, int ldc) {
  return hipsparseSgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
inline hipsparseStatus_t cusparsegemmi(
  hipsparseHandle_t handle, int m, int n, int k, int nnz, const double* alpha,
  const double* A, int lda, const double* cscValB, const int* cscColPtrB,
  const int* cscRowIndB, const double* beta, double* C, int ldc) {
  return hipsparseDgemmi(handle, m, n, k, nnz, alpha, A, lda, cscValB,
                        cscColPtrB, cscRowIndB, beta, C, ldc);
}
/** @} */

};  // namespace Sparse
};  // namespace MLCommon
