# Standalone build of the HugeCTR-on-ROCm runtime smoke test.
# Links against the ported HugeCTR libraries built in ../hugectr_hip/build/lib.
cmake_minimum_required(VERSION 3.21)
project(hugectr_rocm_smoke LANGUAGES CXX)

find_package(HIP REQUIRED)
enable_language(HIP)

if(NOT DEFINED CMAKE_HIP_ARCHITECTURES)
  set(CMAKE_HIP_ARCHITECTURES "gfx950")
endif()

# Treat .cpp as HIP since they contain __device__ lambdas.
set_source_files_properties(smoke_test.cpp     PROPERTIES LANGUAGE HIP)
set_source_files_properties(embedding_test.cpp PROPERTIES LANGUAGE HIP)

set(HUGECTR_HIP_ROOT "${CMAKE_CURRENT_LIST_DIR}/../hugectr_hip")
set(HUGECTR_HIP_BUILD "${HUGECTR_HIP_ROOT}/build")
set(HUGECTR_HIP_LIB "${HUGECTR_HIP_BUILD}/lib")

function(hctr_runtime_test name src)
  add_executable(${name} ${src})
  target_compile_features(${name} PRIVATE cxx_std_17)
  target_compile_definitions(${name} PRIVATE
    HUGECTR_ROCM_PORT
    __HIP_PLATFORM_AMD__
    SPDLOG_HEADER_ONLY
    SPDLOG_FMT_EXTERNAL
    FMT_HEADER_ONLY
  )
  target_include_directories(${name} PRIVATE
    ${HUGECTR_HIP_ROOT}
    ${HUGECTR_HIP_ROOT}/HugeCTR
    ${HUGECTR_HIP_ROOT}/HugeCTR/include
    ${HUGECTR_HIP_ROOT}/third_party
    ${HUGECTR_HIP_ROOT}/third_party/json/single_include
    ${HUGECTR_HIP_ROOT}/third_party/parallel-hashmap
    /opt/rocm/include
    /opt/rocm/include/hipblas
    /opt/rocm/include/hipblaslt
    /opt/rocm/include/hiprand
    /opt/rocm/include/rccl
  )
  target_link_directories(${name} PRIVATE ${HUGECTR_HIP_LIB} /opt/rocm/lib)
  target_link_libraries(${name} PRIVATE
    huge_ctr_shared
    embedding
    hugectr_core23
    gpu_cache
    hipblas
    hipblaslt
    rccl
    hiprand
    rocrand
    amdhip64
    -l:libtbb.so.2
  )
  set_target_properties(${name} PROPERTIES
    BUILD_RPATH "${HUGECTR_HIP_LIB};/opt/rocm/lib"
    INSTALL_RPATH "${HUGECTR_HIP_LIB};/opt/rocm/lib"
  )
endfunction()

hctr_runtime_test(hugectr_rocm_smoke smoke_test.cpp)
hctr_runtime_test(embedding_test     embedding_test.cpp)
