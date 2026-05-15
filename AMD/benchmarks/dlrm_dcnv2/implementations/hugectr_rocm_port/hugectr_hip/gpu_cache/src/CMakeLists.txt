# HIP port: HugeCTR gpu_cache library
cmake_minimum_required(VERSION 3.21)

set(gpu_cache_src
  nv_gpu_cache.cu
  static_table.cu
  # static_hash_table.cu  # ROCm 7.0 amd_hip_fp8.h: __host__ operator float() called
  # uvm_table.cu          # from __device__ contexts. Stubbed below to satisfy the
  static_hash_table_stub.cpp  # linker; unused FP8 paths throw at runtime.
)

foreach(f ${gpu_cache_src})
  set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
endforeach()

add_library(gpu_cache SHARED ${gpu_cache_src})
set_target_properties(gpu_cache PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_features(gpu_cache PUBLIC cxx_std_17)
target_link_libraries(gpu_cache
  PUBLIC
    hip::host
    OpenMP::OpenMP_CXX
)
