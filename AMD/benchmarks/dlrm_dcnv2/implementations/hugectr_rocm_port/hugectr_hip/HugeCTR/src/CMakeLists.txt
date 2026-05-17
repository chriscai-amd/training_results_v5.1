# HIP port: HugeCTR core shared library (huge_ctr_shared) — minimal subset.
# Excludes: GRU, Dropout, BatchNorm, MultiHeadAttention, fused-GEMM-functors (ReLU
# epilogues unsupported in hipBLASLt 1.0), parquet/cudf, multi-node, inference.
cmake_minimum_required(VERSION 3.21)

# tbb: try plain `-ltbb` first (works in containers w/ Ubuntu's libtbb.so symlink).
# Fall back to soname `-l:libtbb.so.2` if only the legacy bare-metal naming exists.
find_library(_HCTR_TBB_PLAIN tbb)
if(_HCTR_TBB_PLAIN)
  set(HCTR_TBB_LINK tbb)
else()
  set(HCTR_TBB_LINK -l:libtbb.so.2)
endif()
message(STATUS "HugeCTR ROCm port: linking against ${HCTR_TBB_LINK}")

file(
  GLOB_RECURSE huge_ctr_src
  RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  *.cpp
  *.cu
  ../embedding_storage/*.cpp
  ../embedding_storage/*.cu
  ../../third_party/dynamic_embedding_table/*.cpp
  ../../third_party/dynamic_embedding_table/*.cu
)

list(FILTER huge_ctr_src EXCLUDE REGEX ".*\\.prehip$")

list(REMOVE_ITEM huge_ctr_src "inference_benchmark/metrics.cpp")
# Note: pybind module sources stay in huge_ctr_shared so the .so already exposes
# the Python helpers; we'll build the actual Python entry point as a separate
# `hugectr` MODULE library (PyO3-style) at the bottom of this file.

# DISABLE_CUDF: drop parquet / cudf code paths.
list(REMOVE_ITEM huge_ctr_src "data_readers/file_source_parquet.cpp")
list(REMOVE_ITEM huge_ctr_src "data_readers/metadata.cpp")
list(REMOVE_ITEM huge_ctr_src "data_readers/parquet_data_reader_worker.cpp")
list(REMOVE_ITEM huge_ctr_src "data_readers/row_group_reading_thread.cpp")
list(REMOVE_ITEM huge_ctr_src "data_readers/dataframe_container.cu")
list(REMOVE_ITEM huge_ctr_src "data_readers/parquet_data_converter.cu")

# Layers using cuDNN (RNN/Dropout/BN) or unsupported cuBLASLt epilogues (RELU_AUX
# family). Not used by DLRM-DCNv2.
list(REMOVE_ITEM huge_ctr_src "layers/gru_layer.cu")
list(REMOVE_ITEM huge_ctr_src "layers/dropout_layer.cu")
list(REMOVE_ITEM huge_ctr_src "layers/batch_norm_layer.cu")
list(REMOVE_ITEM huge_ctr_src "layers/multi_head_attention_layer.cu")
# ROCm port: hipBLASLt 1.2+ provides the RELU_AUX/DRELU/DRELU_BGRAD epilogues,
# so the real fused_gemm_functors.cu is built (was stubbed when only 1.0 was
# available on the bare-metal host).
# interaction_layer.cu uses CUDA WMMA (nvcuda::wmma::*). AMD has equivalent
# MFMA matrix-core intrinsics via rocWMMA, but porting the kernel is a separate
# multi-day task. Excluded from this compile-only port.
list(REMOVE_ITEM huge_ctr_src "layers/interaction_layer.cu")
# Stubs for excluded layers' constructors/methods (referenced by add_dense_layer.cpp).
list(APPEND huge_ctr_src "layers/excluded_layer_stubs.cpp")

# Inference / HPS components are large and depend on Redis/RocksDB; out of scope.
list(FILTER huge_ctr_src EXCLUDE REGEX "inference/.*")
list(FILTER huge_ctr_src EXCLUDE REGEX "inference_benchmark/.*")
list(FILTER huge_ctr_src EXCLUDE REGEX "hps/.*")

# Keep all pybind/* helpers EXCEPT module_main.cpp (which is the PYBIND11_MODULE
# entry point we'll wire into a separate target below).
list(FILTER huge_ctr_src EXCLUDE REGEX "pybind/module_main\\.cpp$")

# dynamic_embedding_table requires the cuCollections submodule (cuco/dynamic_map)
# which we did not vendor in. HierarchicalKV provides the embedding-storage path
# we actually use for DLRM-DCNv2.
list(FILTER huge_ctr_src EXCLUDE REGEX "third_party/dynamic_embedding_table/.*")
list(REMOVE_ITEM huge_ctr_src "../embedding_storage/dynamic_embedding.cu")
# Add the stub so the symbols exist for the dynamic linker.
list(APPEND huge_ctr_src "../embedding_storage/dynamic_embedding_stub.cpp")

# ROCm port Phase 14i.5b (2026-05-13): cktile_mlp_kernel.cu must compile with
# ISOLATED flags (no -include warp-compat.h, no -fopenmp). HCTR's parent flags
# break CK-Tile's WarpGemmDispatcher template instantiation. Build it as a
# separate OBJECT library with cleared COMPILE_OPTIONS, then link into the main
# shared lib. See README §Phase 14i.5.
list(REMOVE_ITEM huge_ctr_src "layers/cktile_mlp_kernel.cu")
set_source_files_properties("layers/cktile_mlp_kernel.cu" PROPERTIES LANGUAGE HIP)
add_library(cktile_mlp_kernel OBJECT "layers/cktile_mlp_kernel.cu")
set_target_properties(cktile_mlp_kernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
# Clear the inherited -include warp-compat + -fopenmp flags. Only set what's
# actually needed for CK-Tile compilation.
set_property(TARGET cktile_mlp_kernel PROPERTY COMPILE_OPTIONS "")
target_compile_options(cktile_mlp_kernel PRIVATE
  $<$<COMPILE_LANGUAGE:HIP>:-O2>
  $<$<COMPILE_LANGUAGE:HIP>:-Wno-unknown-warning-option>
  $<$<COMPILE_LANGUAGE:HIP>:-fno-openmp>
)
target_compile_features(cktile_mlp_kernel PRIVATE cxx_std_17)

# Tag every .cu file as HIP source.
foreach(f ${huge_ctr_src})
  if(f MATCHES "\\.cu$")
    set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

message(STATUS "huge_ctr_shared sources: ${CMAKE_CURRENT_SOURCE_DIR}, count=$<LIST:LENGTH,${huge_ctr_src}>")

add_library(huge_ctr_shared SHARED ${huge_ctr_src} $<TARGET_OBJECTS:cktile_mlp_kernel>)
set_target_properties(huge_ctr_shared PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(huge_ctr_shared
  PUBLIC
    hugectr_core23
    embedding
    gpu_cache
    hip::host
    roc::hipblas
    roc::hipblaslt
    roc::rccl
    roc::rocrand
    hip::hiprand
    OpenMP::OpenMP_CXX
    Threads::Threads
    numa
    stdc++fs
    ${HCTR_TBB_LINK}
  PRIVATE
    aio
)

# Phase 20 (2026-05-16): link librocprofiler-sdk-roctx for HCTR_ROCTX=1
# pipeline-phase tagging in rocprofv3 --marker-trace captures. The lib
# ships with ROCm 7.2.1+ at /opt/rocm/lib; find via name OR fall back
# to direct path with .so.1 suffix (no bare .so symlink in 7.2.1 image).
find_library(HCTR_ROCTX_LIB
  NAMES rocprofiler-sdk-roctx
  PATHS /opt/rocm/lib /opt/rocm-7.2.1/lib
  NO_DEFAULT_PATH)
if(NOT HCTR_ROCTX_LIB)
  if(EXISTS /opt/rocm/lib/librocprofiler-sdk-roctx.so.1)
    set(HCTR_ROCTX_LIB /opt/rocm/lib/librocprofiler-sdk-roctx.so.1)
  endif()
endif()
if(HCTR_ROCTX_LIB)
  message(STATUS "HCTR: linking ROCTX from ${HCTR_ROCTX_LIB}")
  target_link_libraries(huge_ctr_shared PUBLIC ${HCTR_ROCTX_LIB})
  target_include_directories(huge_ctr_shared PUBLIC /opt/rocm/include)
else()
  message(WARNING "HCTR: librocprofiler-sdk-roctx not found; HCTR_ROCTX=1 will fail at runtime")
endif()

target_compile_features(huge_ctr_shared PUBLIC cxx_std_17)

# ============================================================================
# Python module (`hugectr.so`) -- the actual user-facing entry point.
# ============================================================================
# Find Python and pybind11 (installed in /apps/chcai/venvs/cn-py39).
# Python interpreter is set by the parent build (cn-py39 on bare metal,
# /opt/venv/bin/python3 (3.12) inside the rocm/pytorch container). Defer to
# whatever -DPython_EXECUTABLE was passed at cmake configure time.
if(NOT DEFINED CACHE{Python_EXECUTABLE})
  set(Python_EXECUTABLE "/apps/chcai/venvs/cn-py39/bin/python" CACHE FILEPATH "")
endif()
find_package(Python COMPONENTS Interpreter Development REQUIRED)

execute_process(
  COMMAND ${Python_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE pybind11_DIR_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${pybind11_DIR_RAW})
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(hugectr MODULE pybind/module_main.cpp)
target_compile_definitions(hugectr PRIVATE
  HUGECTR_ROCM_PORT
  __HIP_PLATFORM_AMD__
  SPDLOG_HEADER_ONLY
  SPDLOG_FMT_EXTERNAL
  FMT_HEADER_ONLY
  DISABLE_CUDF
)
target_link_libraries(hugectr PRIVATE huge_ctr_shared)
target_compile_features(hugectr PUBLIC cxx_std_17)
set_target_properties(hugectr PROPERTIES
  PREFIX ""
  SUFFIX ".so"
  BUILD_RPATH "${CMAKE_BINARY_DIR}/lib;/opt/rocm/lib"
  INSTALL_RPATH "${CMAKE_BINARY_DIR}/lib;/opt/rocm/lib"
)

