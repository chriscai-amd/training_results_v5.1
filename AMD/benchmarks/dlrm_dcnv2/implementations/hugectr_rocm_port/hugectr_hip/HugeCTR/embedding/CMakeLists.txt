# HIP port: HugeCTR embedding sublibrary
cmake_minimum_required(VERSION 3.21)

file(GLOB embedding_src
  *.cpp
  *.cu
  operators/*.cpp
  operators/*.cu
  data_distributor/*.cpp
  data_distributor/*.cu
  data_distributor/*.cuh
  gpu_barrier/*.cpp
  gpu_barrier/*.cu
)

# Filter out hipify backup files just in case
list(FILTER embedding_src EXCLUDE REGEX ".*\\.prehip$")

# ROCm port: key_filtering_operators.cu now uses hand-written shfl-based tile
# reduce/scan helpers (hctr_rocm_cg::*) in place of cg::reduce/cg::inclusive_scan.

# Mark every .cu/.cuh file as HIP language
foreach(f ${embedding_src})
  if(f MATCHES "\\.cu$|\\.cuh$")
    set_source_files_properties(${f} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()

add_library(embedding SHARED ${embedding_src})

set_target_properties(embedding PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(embedding
  PUBLIC
    hip::host
    roc::rccl
    hugectr_core23
)
target_compile_features(embedding PRIVATE cxx_std_17)
