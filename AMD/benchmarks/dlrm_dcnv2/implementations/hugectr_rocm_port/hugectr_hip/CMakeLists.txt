# HIP/ROCm port of HugeCTR — minimal target, hand-authored for the AMD port attempt.
# Targets: hugectr_core23, embedding, gpu_cache, huge_ctr_shared (no inference/parquet/multinode).

cmake_minimum_required(VERSION 3.21)

project(HugeCTR LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

# --- HIP / ROCm setup -----------------------------------------------------------
find_package(HIP REQUIRED)
enable_language(HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  # MI350X = gfx950
  set(CMAKE_HIP_ARCHITECTURES "gfx950")
endif()
message(STATUS "Target HIP architectures: ${CMAKE_HIP_ARCHITECTURES}")

# Tell CMake to treat .cu files as HIP source — saves having to rename ~600 files.
list(APPEND CMAKE_HIP_SOURCE_FILE_EXTENSIONS cu)

# Required ROCm packages (must be present in /opt/rocm).
find_package(hipblas   REQUIRED CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(hipblaslt REQUIRED CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(rccl      REQUIRED CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(rocrand   REQUIRED CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(hiprand   REQUIRED CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)
find_package(rocthrust QUIET   CONFIG HINTS /opt/rocm /opt/rocm/lib/cmake)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

# Stub out CUDA-only components we removed via post-hipify exclusion.
# (cudnn / cudaToolkit findings were used for sources we no longer build.)

# Disable nvcc-specific things and HugeCTR features that pull CUDA.
set(DISABLE_CUDF ON CACHE BOOL "" FORCE)
set(ENABLE_HDFS  OFF CACHE BOOL "" FORCE)
set(ENABLE_S3    OFF CACHE BOOL "" FORCE)
set(ENABLE_GCS   OFF CACHE BOOL "" FORCE)
set(ENABLE_MULTINODES OFF CACHE BOOL "" FORCE)
set(USE_CUDART_STATIC OFF CACHE BOOL "" FORCE)
set(SHARP_A2A OFF CACHE BOOL "" FORCE)

# Defines used by source code.
add_compile_definitions(
  SPDLOG_HEADER_ONLY
  SPDLOG_FMT_EXTERNAL
  FMT_HEADER_ONLY
  DISABLE_CUDF
  HUGECTR_ROCM_PORT
  __HIP_PLATFORM_AMD__
)

# C++ flags: drop -Werror (hipify residue can warn), keep -Wall.
set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -Wall -Wno-sign-compare -Wno-deprecated-declarations")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -Wall -Wno-sign-compare -Wno-deprecated-declarations")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output dirs
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Generate config.hpp (the autoconf-style header).
configure_file(${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp.in
               ${PROJECT_SOURCE_DIR}/HugeCTR/include/config.hpp)

include_directories(
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/HugeCTR
  ${PROJECT_SOURCE_DIR}/HugeCTR/include
  ${PROJECT_SOURCE_DIR}/third_party
  ${PROJECT_SOURCE_DIR}/third_party/argparse/include
  ${PROJECT_SOURCE_DIR}/third_party/cpptqdm
  ${PROJECT_SOURCE_DIR}/third_party/json/single_include
  ${PROJECT_SOURCE_DIR}/third_party/parallel-hashmap
  ${PROJECT_SOURCE_DIR}/third_party/HierarchicalKV/include
  # ROCm port: cuml replaced by hand-written shim (mlcommon_linalg_hip.cuh)
  ${PROJECT_SOURCE_DIR}/third_party/dynamic_embedding_table
  ${PROJECT_SOURCE_DIR}/gpu_cache/include
  # ROCm port Phase 14i.5 (2026-05-13): CK-Tile fused MLP path requires
  # aiter-bundled composable_kernel (matches example invokers' API), NOT
  # /opt/rocm/include/ck_tile (newer incompatible API; see README §Phase 14i.1).
  # MUST be listed BEFORE /opt/rocm/include so #include "ck_tile/..." resolves
  # to the aiter version. The aiter-bundled CK-Tile (23 MB) was extracted from
  # the docker image at /workspace/aiter/3rdparty/composable_kernel/include/
  # to /home/chcai/aiter_ck_tile on the host, then bind-mounted to /aiter_ck_tile
  # at build time via -v in rebuild_in_container.sh. See README §Phase 14i.5.
  /aiter_ck_tile
  /opt/rocm/include
  /opt/rocm/include/hip
  /opt/rocm/include/hipblas
  /opt/rocm/include/hipblaslt
  /opt/rocm/include/hipblas-common
  /opt/rocm/include/hiprand
  /opt/rocm/include/rocrand
  /opt/rocm/include/rccl
)

link_directories(/opt/rocm/lib)

# Force-include the HIP warp-mask compatibility shims into every HIP source.
# (Provides overloads of __ffs/__popc that accept 64-bit ballot results.)
add_compile_options(
  $<$<COMPILE_LANGUAGE:HIP>:-include${CMAKE_CURRENT_LIST_DIR}/hugectr_hip_warp_compat.h>
)

# ROCm port: amdclang silently no-ops `#pragma omp parallel` without -fopenmp.
# Several HugeCTR .cu files (notably metrics.cu, network.cu, embedding paths)
# use #pragma omp parallel num_threads(NGPU) to drive multi-GPU collectives;
# without this flag those parallel regions execute on the main thread only,
# which causes ncclAllReduce to deadlock waiting on the other ranks. (See
# also HugeCTR/src/CMakeLists.txt for the link-time OpenMP runtime.)
add_compile_options(
  $<$<COMPILE_LANGUAGE:HIP>:-fopenmp>
  $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>
)
add_link_options(-fopenmp)

add_subdirectory(gpu_cache/src)
add_subdirectory(HugeCTR/core23)
add_subdirectory(HugeCTR/embedding)
add_subdirectory(HugeCTR/src)

# Deferred: tools, benchmarks, test, samples, sparse_operation_kit, pybind module.
