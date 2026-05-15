/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
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

#include <hipblas.h>
#include <cuml/common/logger.hpp>
#include <cuml/common/utils.hpp>

namespace MLCommon {
namespace LinAlg {

#define _CUBLAS_ERR_TO_STR(err) \
  case err:                     \
    return #err
inline const char *cublasErr2Str(hipblasStatus_t err) {
  switch (err) {
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_SUCCESS);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_NOT_INITIALIZED);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_ALLOC_FAILED);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_INVALID_VALUE);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_ARCH_MISMATCH);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_MAPPING_ERROR);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_EXECUTION_FAILED);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_INTERNAL_ERROR);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_NOT_SUPPORTED);
    _CUBLAS_ERR_TO_STR(HIPBLAS_STATUS_UNKNOWN);
    default:
      return "CUBLAS_STATUS_UNKNOWN";
  };
}
#undef _CUBLAS_ERR_TO_STR

/** check for cublas runtime API errors and assert accordingly */
#define CUBLAS_CHECK(call)                                         \
  do {                                                             \
    hipblasStatus_t err = call;                                     \
    ASSERT(err == HIPBLAS_STATUS_SUCCESS,                           \
           "CUBLAS call='%s' got errorcode=%d err=%s", #call, err, \
           MLCommon::LinAlg::cublasErr2Str(err));                  \
  } while (0)

/** check for cublas runtime API errors but do not assert */
#define CUBLAS_CHECK_NO_THROW(call)                                          \
  do {                                                                       \
    hipblasStatus_t err = call;                                               \
    if (err != HIPBLAS_STATUS_SUCCESS) {                                      \
      CUML_LOG_ERROR("CUBLAS call='%s' got errorcode=%d err=%s", #call, err, \
                     MLCommon::LinAlg::cublasErr2Str(err));                  \
    }                                                                        \
  } while (0)

/**
 * @defgroup Axpy cublas ax+y operations
 * @{
 */
template <typename T>
hipblasStatus_t cublasaxpy(hipblasHandle_t handle, int n, const T *alpha,
                          const T *x, int incx, T *y, int incy,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublasaxpy(hipblasHandle_t handle, int n,
                                 const float *alpha, const float *x, int incx,
                                 float *y, int incy, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSaxpy(handle, n, alpha, x, incx, y, incy);
}

template <>
inline hipblasStatus_t cublasaxpy(hipblasHandle_t handle, int n,
                                 const double *alpha, const double *x, int incx,
                                 double *y, int incy, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDaxpy(handle, n, alpha, x, incx, y, incy);
}
/** @} */

/**
 * @defgroup gemv cublas gemv calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemv(hipblasHandle_t handle, hipblasOperation_t transA,
                          int m, int n, const T *alfa, const T *A, int lda,
                          const T *x, int incx, const T *beta, T *y, int incy,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemv(hipblasHandle_t handle,
                                 hipblasOperation_t transA, int m, int n,
                                 const float *alfa, const float *A, int lda,
                                 const float *x, int incx, const float *beta,
                                 float *y, int incy, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y,
                     incy);
}

template <>
inline hipblasStatus_t cublasgemv(hipblasHandle_t handle,
                                 hipblasOperation_t transA, int m, int n,
                                 const double *alfa, const double *A, int lda,
                                 const double *x, int incx, const double *beta,
                                 double *y, int incy, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgemv(handle, transA, m, n, alfa, A, lda, x, incx, beta, y,
                     incy);
}
/** @} */

/**
 * @defgroup ger cublas a(x*y.T) + A calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasger(hipblasHandle_t handle, int m, int n, const T *alpha,
                         const T *x, int incx, const T *y, int incy, T *A,
                         int lda, hipStream_t stream);
template <>
inline hipblasStatus_t cublasger(hipblasHandle_t handle, int m, int n,
                                const float *alpha, const float *x, int incx,
                                const float *y, int incy, float *A, int lda,
                                hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}

template <>
inline hipblasStatus_t cublasger(hipblasHandle_t handle, int m, int n,
                                const double *alpha, const double *x, int incx,
                                const double *y, int incy, double *A, int lda,
                                hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDger(handle, m, n, alpha, x, incx, y, incy, A, lda);
}
/** @} */

/**
 * @defgroup gemm cublas gemm calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemm(hipblasHandle_t handle, hipblasOperation_t transA,
                          hipblasOperation_t transB, int m, int n, int k,
                          const T *alfa, const T *A, int lda, const T *B,
                          int ldb, const T *beta, T *C, int ldc,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemm(hipblasHandle_t handle,
                                 hipblasOperation_t transA,
                                 hipblasOperation_t transB, int m, int n, int k,
                                 const float *alfa, const float *A, int lda,
                                 const float *B, int ldb, const float *beta,
                                 float *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb,
                     beta, C, ldc);
}

template <>
inline hipblasStatus_t cublasgemm(hipblasHandle_t handle,
                                 hipblasOperation_t transA,
                                 hipblasOperation_t transB, int m, int n, int k,
                                 const double *alfa, const double *A, int lda,
                                 const double *B, int ldb, const double *beta,
                                 double *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgemm(handle, transA, transB, m, n, k, alfa, A, lda, B, ldb,
                     beta, C, ldc);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemmBatched(hipblasHandle_t handle,
                                 hipblasOperation_t transa,
                                 hipblasOperation_t transb, int m, int n, int k,
                                 const T *alpha, const T *const Aarray[],
                                 int lda, const T *const Barray[], int ldb,
                                 const T *beta, T *Carray[], int ldc,
                                 int batchCount, hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemmBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k, const float *alpha, const float *const Aarray[], int lda,
  const float *const Barray[], int ldb, const float *beta, float *Carray[],
  int ldc, int batchCount, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}

template <>
inline hipblasStatus_t cublasgemmBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k, const double *alpha, const double *const Aarray[],
  int lda, const double *const Barray[], int ldb, const double *beta,
  double *Carray[], int ldc, int batchCount, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}
/** @} */

/**
 * @defgroup gemmbatched cublas gemmbatched calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgemmStridedBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k, const T *alpha, const T *const Aarray, int lda,
  long long int strideA, const T *const Barray, int ldb, long long int strideB,
  const T *beta, T *Carray, int ldc, long long int strideC, int batchCount,
  hipStream_t stream);

template <>
inline hipblasStatus_t cublasgemmStridedBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k, const float *alpha, const float *const Aarray, int lda,
  long long int strideA, const float *const Barray, int ldb,
  long long int strideB, const float *beta, float *Carray, int ldc,
  long long int strideC, int batchCount, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                   Aarray, lda, strideA, Barray, ldb, strideB,
                                   beta, Carray, ldc, strideC, batchCount);
}

template <>
inline hipblasStatus_t cublasgemmStridedBatched(
  hipblasHandle_t handle, hipblasOperation_t transa, hipblasOperation_t transb,
  int m, int n, int k, const double *alpha, const double *const Aarray, int lda,
  long long int strideA, const double *const Barray, int ldb,
  long long int strideB, const double *beta, double *Carray, int ldc,
  long long int strideC, int batchCount, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgemmStridedBatched(handle, transa, transb, m, n, k, alpha,
                                   Aarray, lda, strideA, Barray, ldb, strideB,
                                   beta, Carray, ldc, strideC, batchCount);
}
/** @} */

/**
 * @defgroup solverbatched cublas getrf/gettribatched calls
 * @{
 */

template <typename T>
hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle, int n,
                                  T *const A[],    /*Device pointer*/
                                  int lda, int *P, /*Device Pointer*/
                                  int *info,       /*Device Pointer*/
                                  int batchSize, hipStream_t stream);

template <>
inline hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle, int n,
                                         float *const A[], /*Device pointer*/
                                         int lda, int *P,  /*Device Pointer*/
                                         int *info,        /*Device Pointer*/
                                         int batchSize, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <>
inline hipblasStatus_t cublasgetrfBatched(hipblasHandle_t handle, int n,
                                         double *const A[], /*Device pointer*/
                                         int lda, int *P,   /*Device Pointer*/
                                         int *info,         /*Device Pointer*/
                                         int batchSize, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
}

template <typename T>
hipblasStatus_t cublasgetriBatched(hipblasHandle_t handle, int n,
                                  const T *const A[],    /*Device pointer*/
                                  int lda, const int *P, /*Device pointer*/
                                  T *const C[],          /*Device pointer*/
                                  int ldc, int *info, int batchSize,
                                  hipStream_t stream);

template <>
inline hipblasStatus_t cublasgetriBatched(
  hipblasHandle_t handle, int n, const float *const A[], /*Device pointer*/
  int lda, const int *P,                                /*Device pointer*/
  float *const C[],                                     /*Device pointer*/
  int ldc, int *info, int batchSize, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

template <>
inline hipblasStatus_t cublasgetriBatched(
  hipblasHandle_t handle, int n, const double *const A[], /*Device pointer*/
  int lda, const int *P,                                 /*Device pointer*/
  double *const C[],                                     /*Device pointer*/
  int ldc, int *info, int batchSize, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgetriBatched(handle, n, A, lda, P, C, ldc, info, batchSize);
}

/** @} */

/**
 * @defgroup gelsbatched cublas gelsbatched calls
 * @{
 */

template <typename T>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,
                                        hipblasOperation_t trans, int m, int n,
                                        int nrhs, T *Aarray[], int lda,
                                        T *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        hipStream_t stream = 0);

template <>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,
                                        hipblasOperation_t trans, int m, int n,
                                        int nrhs, float *Aarray[], int lda,
                                        float *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
}

template <>
inline hipblasStatus_t cublasgelsBatched(hipblasHandle_t handle,
                                        hipblasOperation_t trans, int m, int n,
                                        int nrhs, double *Aarray[], int lda,
                                        double *Carray[], int ldc, int *info,
                                        int *devInfoArray, int batchSize,
                                        hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgelsBatched(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
}

/** @} */

/**
 * @defgroup geam cublas geam calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasgeam(hipblasHandle_t handle, hipblasOperation_t transA,
                          hipblasOperation_t transB, int m, int n, const T *alfa,
                          const T *A, int lda, const T *beta, const T *B,
                          int ldb, T *C, int ldc, hipStream_t stream);

template <>
inline hipblasStatus_t cublasgeam(hipblasHandle_t handle,
                                 hipblasOperation_t transA,
                                 hipblasOperation_t transB, int m, int n,
                                 const float *alfa, const float *A, int lda,
                                 const float *beta, const float *B, int ldb,
                                 float *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb,
                     C, ldc);
}

template <>
inline hipblasStatus_t cublasgeam(hipblasHandle_t handle,
                                 hipblasOperation_t transA,
                                 hipblasOperation_t transB, int m, int n,
                                 const double *alfa, const double *A, int lda,
                                 const double *beta, const double *B, int ldb,
                                 double *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDgeam(handle, transA, transB, m, n, alfa, A, lda, beta, B, ldb,
                     C, ldc);
}
/** @} */

/**
 * @defgroup symm cublas symm calls
 * @{
 */
template <typename T>
hipblasStatus_t cublassymm(hipblasHandle_t handle, hipblasSideMode_t side,
                          hipblasFillMode_t uplo, int m, int n, const T *alpha,
                          const T *A, int lda, const T *B, int ldb,
                          const T *beta, T *C, int ldc, hipStream_t stream);

template <>
inline hipblasStatus_t cublassymm(hipblasHandle_t handle, hipblasSideMode_t side,
                                 hipblasFillMode_t uplo, int m, int n,
                                 const float *alpha, const float *A, int lda,
                                 const float *B, int ldb, const float *beta,
                                 float *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}

template <>
inline hipblasStatus_t cublassymm(hipblasHandle_t handle, hipblasSideMode_t side,
                                 hipblasFillMode_t uplo, int m, int n,
                                 const double *alpha, const double *A, int lda,
                                 const double *B, int ldb, const double *beta,
                                 double *C, int ldc, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDsymm(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C,
                     ldc);
}
/** @} */

/**
 * @defgroup syrk cublas syrk calls
 * @{
 */
template <typename T>
hipblasStatus_t cublassyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                          hipblasOperation_t trans, int n, int k, const T *alpha,
                          const T *A, int lda, const T *beta, T *C, int ldc,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublassyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                                 hipblasOperation_t trans, int n, int k,
                                 const float *alpha, const float *A, int lda,
                                 const float *beta, float *C, int ldc,
                                 hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}

template <>
inline hipblasStatus_t cublassyrk(hipblasHandle_t handle, hipblasFillMode_t uplo,
                                 hipblasOperation_t trans, int n, int k,
                                 const double *alpha, const double *A, int lda,
                                 const double *beta, double *C, int ldc,
                                 hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDsyrk(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
}
/** @} */

/**
 * @defgroup nrm2 cublas nrm2 calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasnrm2(hipblasHandle_t handle, int n, const T *x, int incx,
                          T *result, hipStream_t stream);

template <>
inline hipblasStatus_t cublasnrm2(hipblasHandle_t handle, int n, const float *x,
                                 int incx, float *result, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSnrm2(handle, n, x, incx, result);
}

template <>
inline hipblasStatus_t cublasnrm2(hipblasHandle_t handle, int n, const double *x,
                                 int incx, double *result,
                                 hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDnrm2(handle, n, x, incx, result);
}
/** @} */

template <typename T>
hipblasStatus_t cublastrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                          hipblasFillMode_t uplo, hipblasOperation_t trans,
                          hipblasDiagType_t diag, int m, int n, const T *alpha,
                          const T *A, int lda, T *B, int ldb,
                          hipStream_t stream);

template <>
inline hipblasStatus_t cublastrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                                 hipblasFillMode_t uplo, hipblasOperation_t trans,
                                 hipblasDiagType_t diag, int m, int n,
                                 const float *alpha, const float *A, int lda,
                                 float *B, int ldb, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasStrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

template <>
inline hipblasStatus_t cublastrsm(hipblasHandle_t handle, hipblasSideMode_t side,
                                 hipblasFillMode_t uplo, hipblasOperation_t trans,
                                 hipblasDiagType_t diag, int m, int n,
                                 const double *alpha, const double *A, int lda,
                                 double *B, int ldb, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDtrsm(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B,
                     ldb);
}

/**
 * @defgroup dot cublas dot calls
 * @{
 */
template <typename T>
hipblasStatus_t cublasdot(hipblasHandle_t handle, int n, const T *x, int incx,
                         const T *y, int incy, T *result, hipStream_t stream);

template <>
inline hipblasStatus_t cublasdot(hipblasHandle_t handle, int n, const float *x,
                                int incx, const float *y, int incy,
                                float *result, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasSdot(handle, n, x, incx, y, incy, result);
}

template <>
inline hipblasStatus_t cublasdot(hipblasHandle_t handle, int n, const double *x,
                                int incx, const double *y, int incy,
                                double *result, hipStream_t stream) {
  CUBLAS_CHECK(hipblasSetStream(handle, stream));
  return hipblasDdot(handle, n, x, incx, y, incy, result);
}
/** @} */

};  // namespace LinAlg
};  // namespace MLCommon
