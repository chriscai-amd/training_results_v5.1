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

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
// /* HugeCTR ROCm port: <hipDNN.h> removed */          // ROCm port: cuDNN/MIOpen excluded (no cuDNN-using layers built)
#include <hiprand/hiprand.h>
#include <rccl/rccl.h>
// /* HugeCTR ROCm port: <nvml.h> removed */            // ROCm port: NVML has no AMD equivalent; calls stubbed

#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef HCTR_CODE_LOCATION_
#error HCTR_CODE_LOCATION_ already defined. Potential naming conflict!
#endif
#define HCTR_CODE_LOCATION_() \
  HugeCTR::core23::CodeReference { __FILE__, __LINE__, __func__, nullptr }

#ifdef HCTR_CODE_REFERENCE_
#error HCTR_CODE_REFERENCE_ already defined. Potential naming conflict!
#endif
#define HCTR_CODE_REFERENCE_(EXPR) \
  HugeCTR::core23::CodeReference { __FILE__, __LINE__, __func__, #EXPR }

/**
 * For HugeCTR own error types, it is up to users to define the msesage.
 */
#ifdef HCTR_OWN_THROW_
#error HCTR_OWN_THROW_ already defined. Potential naming conflict!
#endif
#define HCTR_OWN_THROW_(EXPR, ...)                                                           \
  do {                                                                                       \
    const HugeCTR::Error_t _expr_eval = (EXPR);                                              \
    if (_expr_eval != HugeCTR::Error_t::Success) {                                           \
      throw HugeCTR::core23::RuntimeError(_expr_eval, HCTR_CODE_REFERENCE_(EXPR),            \
                                          HugeCTR::core23::hctr_render_string(__VA_ARGS__)); \
    }                                                                                        \
  } while (0)

/**
 * For third-party library calls such as CUDA, cuBLAS and NCCL, use this macro.
 */
#ifdef HCTR_LIB_THROW_
#error HCTR_LIB_THROW_ already defined. Potential naming conflict!
#endif
#define HCTR_LIB_THROW_(EXPR)                                                            \
  do {                                                                                   \
    const auto _expr_eval = (EXPR);                                                      \
    const auto _expr_eval_err = HugeCTR::core23::to_error(_expr_eval);                   \
    if (_expr_eval_err != HugeCTR::Error_t::Success) {                                   \
      throw HugeCTR::core23::RuntimeError(_expr_eval_err, HCTR_CODE_REFERENCE_(EXPR),    \
                                          HugeCTR::core23::to_error_string(_expr_eval)); \
    }                                                                                    \
  } while (0)

/**
 * Because MPI error code is `int`, it is safe to have a separate macro for MPI,
 * rather than reserving `int` as MPI error type. We don't want this set of macros
 * to become another source of errors.
 */
#ifdef HCTR_MPI_THROW_
#error HCTR_MPI_THROW_ already defined. Potential naming conflict!
#endif
#define HCTR_MPI_THROW_(EXPR)                                          \
  do {                                                                 \
    int _expr_eval = (EXPR);                                           \
    if (_expr_eval != MPI_SUCCESS) {                                   \
      char msg_buf[MPI_MAX_ERROR_STRING];                              \
      int msg_len{MPI_MAX_ERROR_STRING};                               \
      _expr_eval = MPI_Error_string(_expr_eval, msg_buf, &msg_len);    \
      throw HugeCTR::core23::RuntimeError(                             \
          HugeCTR::Error_t::MpiError, HCTR_CODE_REFERENCE_(EXPR),      \
          _expr_eval == MPI_SUCCESS ? msg_buf : "Unknown MPI error!"); \
    }                                                                  \
  } while (0)

/**
 * Macro to emit an exception if the supplied expression does not evaluate true.
 */
#ifdef HCTR_THROW_IF_
#error HCTR_THROW_IF_ already defined. Potential naming conflict!
#endif
#define HCTR_THROW_IF_(EXPR, ERROR, ...)                                                     \
  do {                                                                                       \
    if ((EXPR)) {                                                                            \
      throw HugeCTR::core23::RuntimeError((ERROR), HCTR_CODE_REFERENCE_(EXPR),               \
                                          HugeCTR::core23::hctr_render_string(__VA_ARGS__)); \
    }                                                                                        \
  } while (0)

/**
 * Legacy macros.
 */
#ifdef HCTR_LOCATION
#error HCTR_LOCATION already defined. Potential naming conflict!
#endif
#define HCTR_LOCATION() HCTR_CODE_LOCATION_()

#ifdef HCTR_OWN_THROW
#error HCTR_OWN_THROW already defined. Potential naming conflict!
#endif
#define HCTR_OWN_THROW(EXPR, ...) HCTR_OWN_THROW_(EXPR, __VA_ARGS__)

#ifdef HCTR_LIB_THROW
#error HCTR_LIB_THROW already defined. Potential naming conflict!
#endif
#define HCTR_LIB_THROW(EXPR) HCTR_LIB_THROW_(EXPR)

#ifdef HCTR_MPI_THROW
#error HCTR_MPI_THROW already defined. Potential naming conflict!
#endif
#define HCTR_MPI_THROW(EXPR) HCTR_MPI_THROW_(EXPR)

#ifdef HCTR_THROW_IF
#error HCTR_THROW_IF already defined. Potential naming conflict!
#endif
#define HCTR_THROW_IF(EXPR, ERROR, ...) HCTR_THROW_IF_(EXPR, ERROR, __VA_ARGS__)

namespace HugeCTR {

enum class Error_t {
  // -- HCTR errors ---
  Success,
  FileCannotOpen,
  BrokenFile,
  OutOfMemory,
  OutOfBound,
  WrongInput,
  IllegalCall,
  Deprecated,
  NotInitialized,
  UnSupportedFormat,
  InvalidEnv,
  DataCheckError,
  UnspecificError,
  EndOfFile,
  // -- MPI errors --
  MpiError,
  // -- CUDA / NVIDIA library errors ---
  CudaDriverError,
  CudaRuntimeError,
  CublasError,
  CudnnError,
  CurandError,
  NcclError,
  NvmlError,
};

namespace core23 {

struct CodeReference {
  const char* const file;
  const size_t line;
  const char* function;
  const char* expression;
};

inline std::ostream& operator<<(std::ostream& os, const CodeReference& ref) {
  if (ref.expression) {
    os << ref.expression << ' ';
  }
  os << '(' << ref.function << " @ " << ref.file << ':' << ref.line << ')';
  return os;
}

/**
 * An internal exception, as a child of std::runtime_error, to carry the error code.
 */
class RuntimeError : public std::runtime_error {
 public:
  RuntimeError() = delete;

  inline RuntimeError(const Error_t err, const std::string& what)
      : std::runtime_error(what), error{err} {}

  inline RuntimeError(const Error_t err, const CodeReference& ref, const std::string& msg)
      : RuntimeError(err, ref, msg.c_str()) {}

  inline RuntimeError(const Error_t err, const CodeReference& ref, const char* const msg)
      : RuntimeError(err, [ref, msg]() {
          std::ostringstream os;
          os << "Runtime error: " << msg << "\n\t" << ref;
          return os.str();
        }()) {}

  /**
   * Get the error code from exception.
   * @return error
   **/
  const Error_t error;
};

/**
 * Map HugeCTR runtime error to HugeCTR error code.
 */
[[nodiscard]] inline Error_t to_error(const RuntimeError& r) { return r.error; }

/**
 * Map arbitrary exception to HugeCTR error code.
 */
[[nodiscard]] inline Error_t to_error(const std::exception&) { return Error_t::UnspecificError; }

/**
 * Map CUDA driver/runtime error codes to HugeCTR error codes.
 * On HIP, both CUDA driver `CUresult` and runtime `cudaError_t` collapse into the
 * single `hipError_t` type, so the two CUDA-side overloads are merged here.
 */
[[nodiscard]] inline Error_t to_error(const hipError_t r) {
  return r == hipSuccess ? Error_t::Success : Error_t::CudaRuntimeError;
}

/**
 * Map cuBLAS error codes to HugeCTR error codes.
 */
[[nodiscard]] inline Error_t to_error(const hipblasStatus_t s) {
  return s == HIPBLAS_STATUS_SUCCESS ? Error_t::Success : Error_t::CublasError;
}

// HugeCTR ROCm port: cuDNN/MIOpen excluded; to_error(hipdnnStatus_t) removed.

/**
 * Map cuRAND error codes to HugeCTR error codes.
 */
[[nodiscard]] inline Error_t to_error(const hiprandStatus_t s) {
  return (s == HIPRAND_STATUS_SUCCESS) ? Error_t::Success : Error_t::CurandError;
}

/**
 * Map NCCL error codes to HugeCTR error codes.
 */
[[nodiscard]] inline Error_t to_error(const ncclResult_t r) {
  return r == ncclSuccess ? Error_t::Success : Error_t::NcclError;
}

// HugeCTR ROCm port: NVML excluded; to_error(nvmlReturn_t) removed.

/**
 * Map HIP error codes to string. (CUDA driver+runtime collapsed in HIP.)
 */
[[nodiscard]] inline const char* to_error_string(const hipError_t e) {
  return hipGetErrorString(e);
}

/**
 * Map cuBLAS error codes to string.
 */
[[nodiscard]] inline const char* to_error_string(const hipblasStatus_t s) {
  switch (s) {
    case HIPBLAS_STATUS_SUCCESS:
      return "cuBLAS operation completed successfully.";
    case HIPBLAS_STATUS_NOT_INITIALIZED:
      return "The cuBLAS library was not initialized.";
    case HIPBLAS_STATUS_ALLOC_FAILED:
      return "Resource allocation failed inside the cuBLAS library.";
    case HIPBLAS_STATUS_INVALID_VALUE:
      return "An unsupported value or parameter was passed to the cuBLAS function.";
    case HIPBLAS_STATUS_ARCH_MISMATCH:
      return "The function requires a feature absent from the device architecture.";
    case HIPBLAS_STATUS_MAPPING_ERROR:
      return "An access to GPU memory space failed.";
    case HIPBLAS_STATUS_EXECUTION_FAILED:
      return "The GPU program failed to execute.";
    case HIPBLAS_STATUS_INTERNAL_ERROR:
      return "An internal cuBLAS operation failed.";
    case HIPBLAS_STATUS_NOT_SUPPORTED:
      return "The functionality requested is not supported.";
    case HIPBLAS_STATUS_UNKNOWN:
      return "The functionality requested requires some license and an error was detected when "
             "trying to check the current licensing.";
    default:
      return "Unknown cuBLAS error.";
  }
}

// HugeCTR ROCm port: cuDNN/MIOpen excluded; to_error_string(hipdnnStatus_t) removed.

/**
 * Map cuRAND error codes to string.
 */
[[nodiscard]] inline const char* to_error_string(const hiprandStatus_t s) {
  switch (s) {
    case HIPRAND_STATUS_SUCCESS:
      return "No errors.";
    case HIPRAND_STATUS_VERSION_MISMATCH:
      return "Header file and linked library version do not match.";
    case HIPRAND_STATUS_NOT_INITIALIZED:
      return "Generator not initialized.";
    case HIPRAND_STATUS_ALLOCATION_FAILED:
      return "Memory allocation failed.";
    case HIPRAND_STATUS_TYPE_ERROR:
      return "Generator is wrong type.";
    case HIPRAND_STATUS_OUT_OF_RANGE:
      return "Argument out of range.";
    case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
      return "Length requested is not a multiple of dimension.";
    case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
      return "GPU does not have double precision required by MRG32k3a.";
    case HIPRAND_STATUS_LAUNCH_FAILURE:
      return "Kernel launch failure.";
    case HIPRAND_STATUS_PREEXISTING_FAILURE:
      return "Preexisting failure on library entry.";
    case HIPRAND_STATUS_INITIALIZATION_FAILED:
      return "Initialization of CUDA failed.";
    case HIPRAND_STATUS_ARCH_MISMATCH:
      return "Architecture mismatch, GPU does not support requested feature.";
    case HIPRAND_STATUS_INTERNAL_ERROR:
      return "Internal library error.";
    default:
      return "Unknown cuRAND error.";
  }
}

/**
 * Map NCCL error codes to string.
 */
[[nodiscard]] inline const char* to_error_string(const ncclResult_t r) {
  return ncclGetErrorString(r);
}

// HugeCTR ROCm port: NVML excluded; to_error_string(nvmlReturn_t) removed.

[[nodiscard]] inline const char* hctr_render_string() { return ""; }

[[nodiscard]] inline const char* hctr_render_string(const char* const arg0) { return arg0; }

[[nodiscard]] inline const std::string& hctr_render_string(const std::string& arg0) { return arg0; }

template <typename Arg0>
[[nodiscard]] inline std::string hctr_render_string(const Arg0& arg0) {
  std::ostringstream os;
  os << arg0;
  return os.str();
}

template <typename Arg0, typename... Args>
[[nodiscard]] inline std::string hctr_render_string(const Arg0& arg0, Args&&... args) {
  std::ostringstream os;
  os << arg0;
  (os << ... << args);
  return os.str();
}

}  // namespace core23
}  // namespace HugeCTR
